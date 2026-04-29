#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "VulkanEngine.hpp"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_INCLUDE_JSON
#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include "tiny_gltf.h"
#include <glm/gtc/type_ptr.hpp>
#include "SceneSerializer.hpp"
#include "ProjectManager.hpp"
#include "EngineLog.hpp"
#include "AssetPipeline.hpp"
#include <filesystem>
#include <stdexcept>
#include <unordered_map>
#include <fstream>

// ---------------------------------------------------------------------------
// OBJ mesh loader — shared by loadMesh() and packAssets()
// ---------------------------------------------------------------------------
static void loadMeshDataFromOBJ(const std::string& path,
                                 std::vector<Vertex>& outVerts,
                                 std::vector<uint32_t>& outIdx) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> mats;
    std::string warn, err;
    if (!tinyobj::LoadObj(&attrib, &shapes, &mats, &warn, &err, path.c_str()))
        THROW_ENGINE_ERROR("Eroare: Nu am putut incarca " + path + " " + warn + err);
    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex v{};
            v.pos   = { attrib.vertices[3*index.vertex_index+0],
                        attrib.vertices[3*index.vertex_index+1],
                        attrib.vertices[3*index.vertex_index+2] };
            v.color = {1.0f, 1.0f, 1.0f};
            if (index.normal_index >= 0)
                v.normal = { attrib.normals[3*index.normal_index+0],
                             attrib.normals[3*index.normal_index+1],
                             attrib.normals[3*index.normal_index+2] };
            if (index.texcoord_index >= 0)
                v.texCoord = { attrib.texcoords[2*index.texcoord_index+0],
                               1.0f - attrib.texcoords[2*index.texcoord_index+1] };
            outIdx.push_back((uint32_t)outVerts.size());
            outVerts.push_back(v);
        }
    }
}

// ---------------------------------------------------------------------------
// Vertex-clustering mesh simplifier
// ratio: target fraction of original vertex count (e.g. 0.5 = 50%)
// ---------------------------------------------------------------------------
struct SimpleMeshData { std::vector<Vertex> verts; std::vector<uint32_t> idx; };

static SimpleMeshData simplifyMeshVC(const std::vector<Vertex>& verts,
                                     const std::vector<uint32_t>& idx,
                                     float ratio) {
    if (verts.size() < 12 || idx.size() < 9)
        return {verts, idx};

    glm::vec3 mn = verts[0].pos, mx = verts[0].pos;
    for (const auto& v : verts) { mn = glm::min(mn, v.pos); mx = glm::max(mx, v.pos); }
    glm::vec3 range = mx - mn + glm::vec3(1e-6f);

    int N = std::max(2, (int)std::round(std::cbrt(ratio * (float)verts.size())));

    auto cellKey = [&](const glm::vec3& p) -> uint64_t {
        int cx = std::clamp((int)((p.x - mn.x) / range.x * N), 0, N-1);
        int cy = std::clamp((int)((p.y - mn.y) / range.y * N), 0, N-1);
        int cz = std::clamp((int)((p.z - mn.z) / range.z * N), 0, N-1);
        return (uint64_t)cx * N*N + (uint64_t)cy * N + cz;
    };

    struct CellAcc { glm::vec3 pos{}, normal{}, color{}; glm::vec2 uv{}; int n = 0; };
    std::unordered_map<uint64_t, CellAcc> cells;
    for (const auto& v : verts) {
        auto& c = cells[cellKey(v.pos)];
        c.pos += v.pos; c.normal += v.normal; c.color += v.color; c.uv += v.texCoord; c.n++;
    }

    std::unordered_map<uint64_t, uint32_t> keyToIdx;
    std::vector<Vertex> outVerts;
    outVerts.reserve(cells.size());
    for (auto& [k, c] : cells) {
        float f = 1.0f / c.n;
        Vertex v{};
        v.pos      = c.pos * f;
        v.normal   = glm::length(c.normal) > 0.0f ? glm::normalize(c.normal) : glm::vec3(0,1,0);
        v.color    = c.color * f;
        v.texCoord = c.uv * f;
        keyToIdx[k] = (uint32_t)outVerts.size();
        outVerts.push_back(v);
    }

    std::vector<uint32_t> outIdx;
    outIdx.reserve(idx.size());
    for (size_t t = 0; t + 2 < idx.size(); t += 3) {
        auto k0 = cellKey(verts[idx[t  ]].pos);
        auto k1 = cellKey(verts[idx[t+1]].pos);
        auto k2 = cellKey(verts[idx[t+2]].pos);
        if (k0 == k1 || k1 == k2 || k0 == k2) continue;
        outIdx.push_back(keyToIdx[k0]);
        outIdx.push_back(keyToIdx[k1]);
        outIdx.push_back(keyToIdx[k2]);
    }

    return {outVerts, outIdx};
}

// Upload a CPU mesh as LOD VB+IB (no RT flags needed for LODs)
static void uploadLODBuffers(VkDevice device,
                             const std::function<void(VkDeviceSize,VkBufferUsageFlags,
                                 VkMemoryPropertyFlags,VkBuffer&,VkDeviceMemory&,bool)>& createBuffer,
                             const std::function<void(VkBuffer,VkBuffer,VkDeviceSize)>& copyBuffer,
                             const std::vector<Vertex>& verts, const std::vector<uint32_t>& indices,
                             VkBuffer& outVB, VkDeviceMemory& outVBMem,
                             VkBuffer& outIB, VkDeviceMemory& outIBMem,
                             uint32_t& outIdxCount) {
    if (verts.empty() || indices.empty()) return;
    outIdxCount = (uint32_t)indices.size();

    VkDeviceSize vs = sizeof(Vertex) * verts.size();
    VkBuffer vStage; VkDeviceMemory vStageM;
    createBuffer(vs, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        vStage, vStageM, false);
    void* vd; vkMapMemory(device, vStageM, 0, vs, 0, &vd);
    memcpy(vd, verts.data(), vs); vkUnmapMemory(device, vStageM);
    createBuffer(vs, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outVB, outVBMem, false);
    copyBuffer(vStage, outVB, vs);
    vkDestroyBuffer(device, vStage, nullptr); vkFreeMemory(device, vStageM, nullptr);

    VkDeviceSize is = sizeof(uint32_t) * indices.size();
    VkBuffer iStage; VkDeviceMemory iStageM;
    createBuffer(is, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        iStage, iStageM, false);
    void* id; vkMapMemory(device, iStageM, 0, is, 0, &id);
    memcpy(id, indices.data(), is); vkUnmapMemory(device, iStageM);
    createBuffer(is, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outIB, outIBMem, false);
    copyBuffer(iStage, outIB, is);
    vkDestroyBuffer(device, iStage, nullptr); vkFreeMemory(device, iStageM, nullptr);
}

void VulkanEngine::scanProjects() {
    ProjectManager::scanProjects(workspace);
}

void VulkanEngine::initWorkspace() {
    ProjectManager::initWorkspace(workspace);
}

void VulkanEngine::saveScene(const std::string& filename) {
    // filename already includes .scene extension
    SceneSerializer::save(activeScene, workspace.activeProject.getScenesPath() + "/" + filename);
    LOG_INFO("Scena salvata: %s", filename.c_str());
}

void VulkanEngine::loadScene(const std::string& filename) {
    vkDeviceWaitIdle(renderer.device);
    for (auto& [id, es] : renderer.particleEmitters) renderer.destroyEmitterState(es);
    renderer.particleEmitters.clear();
    { std::vector<uint32_t> ids; for (auto& [id, _] : renderer.skinInstances) ids.push_back(id);
      for (auto id : ids) renderer.destroySkinInstance(id); }
    runtime.clearSceneState();
    activeScene = SceneSerializer::load(workspace.activeProject.getScenesPath() + "/" + filename);
    LOG_INFO("Scena incarcata: %s (%d entitati)", filename.c_str(), (int)activeScene.entities.size());
}

void VulkanEngine::switchProject(const std::string& projectName) {
    vkDeviceWaitIdle(renderer.device);
    for (auto& [id, es] : renderer.particleEmitters) renderer.destroyEmitterState(es);
    renderer.particleEmitters.clear();
    { std::vector<uint32_t> ids; for (auto& [id, _] : renderer.skinInstances) ids.push_back(id);
      for (auto id : ids) renderer.destroySkinInstance(id); }
    runtime.clearSceneState();
    for (auto& [name, mesh] : renderer.loadedMeshes) {
        vkDestroyBuffer(renderer.device, mesh.vertexBuffer, nullptr);
        vkFreeMemory(renderer.device, mesh.vertexBufferMemory, nullptr);
        vkDestroyBuffer(renderer.device, mesh.indexBuffer, nullptr);
        vkFreeMemory(renderer.device, mesh.indexBufferMemory, nullptr);
    }
    renderer.loadedMeshes.clear();
    for (auto& [name, smr] : renderer.loadedSkinnedMeshes) {
        vkDestroyBuffer(renderer.device, smr.vertexBuffer, nullptr);
        vkFreeMemory(renderer.device, smr.vertexBufferMemory, nullptr);
        vkDestroyBuffer(renderer.device, smr.indexBuffer, nullptr);
        vkFreeMemory(renderer.device, smr.indexBufferMemory, nullptr);
    }
    renderer.loadedSkinnedMeshes.clear();
    renderer.loadedSkins.clear();
    ProjectManager::applyProject(projectName, workspace);
    renderer.scanModelsFolder();
    renderer.scanTexturesFolder();
    renderer.scanAudioFolder();
    renderer.reloadProjectTextures();
    renderer.selectedModelIndex = 0;
    renderer.selectedTextureIndex = 0;
    activeScene.entities.clear();
}

void VulkanEngine::createNewProject(const std::string& projectName) {
    if (projectName.empty()) return;
    ProjectManager::createProjectFolders(projectName);
    scanProjects();
    switchProject(projectName);
}

void Renderer::scanModelsFolder() {
    availableModels.clear();
    std::string path = workspace->activeProject.getModelsPath();

    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            auto ext = entry.path().extension();
            if (ext == ".obj" || ext == ".gltf" || ext == ".glb") {
                availableModels.push_back(entry.path().filename().string());
            }
        }
    }
}

// Extract all primitives from a GLTF/GLB file into flat vertex+index lists.
// Handles packed and strided buffer views, uint8/uint16/uint32 index types.
static void loadMeshDataFromGLTF(const std::string& path,
                                  std::vector<Vertex>& outVerts,
                                  std::vector<uint32_t>& outIdx)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;

    bool ok = (path.size() >= 4 && path.substr(path.size() - 4) == ".glb")
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!ok)
        THROW_ENGINE_ERROR("GLTF load error: " + err);

    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) continue;

            auto fetchAccessorBase = [&](int accIdx, size_t& stride, size_t elemSize) -> const uint8_t* {
                const auto& acc = model.accessors[accIdx];
                const auto& bv  = model.bufferViews[acc.bufferView];
                stride = bv.byteStride > 0 ? bv.byteStride : elemSize;
                return model.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset;
            };

            // --- Positions (required) ---
            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end()) continue;
            size_t posStride;
            const uint8_t* posBase = fetchAccessorBase(posIt->second, posStride, sizeof(float) * 3);
            const size_t vertCount = model.accessors[posIt->second].count;

            // --- Normals (optional) ---
            size_t normStride = 0;
            const uint8_t* normBase = nullptr;
            auto normIt = prim.attributes.find("NORMAL");
            if (normIt != prim.attributes.end())
                normBase = fetchAccessorBase(normIt->second, normStride, sizeof(float) * 3);

            // --- Texcoords (optional) ---
            size_t uvStride = 0;
            const uint8_t* uvBase = nullptr;
            auto uvIt = prim.attributes.find("TEXCOORD_0");
            if (uvIt != prim.attributes.end())
                uvBase = fetchAccessorBase(uvIt->second, uvStride, sizeof(float) * 2);

            uint32_t indexOffset = (uint32_t)outVerts.size();

            for (size_t i = 0; i < vertCount; i++) {
                Vertex v{};
                const float* p = reinterpret_cast<const float*>(posBase + i * posStride);
                v.pos   = { p[0], p[1], p[2] };
                v.color = { 1.0f, 1.0f, 1.0f };
                if (normBase) {
                    const float* n = reinterpret_cast<const float*>(normBase + i * normStride);
                    v.normal = { n[0], n[1], n[2] };
                }
                if (uvBase) {
                    const float* uv = reinterpret_cast<const float*>(uvBase + i * uvStride);
                    v.texCoord = { uv[0], uv[1] };
                }
                outVerts.push_back(v);
            }

            // --- Indices ---
            if (prim.indices >= 0) {
                const auto& idxAcc = model.accessors[prim.indices];
                const auto& idxBV  = model.bufferViews[idxAcc.bufferView];
                const uint8_t* idxBase = model.buffers[idxBV.buffer].data.data()
                                       + idxBV.byteOffset + idxAcc.byteOffset;
                for (size_t i = 0; i < idxAcc.count; i++) {
                    uint32_t idx = 0;
                    switch (idxAcc.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            idx = idxBase[i]; break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            idx = reinterpret_cast<const uint16_t*>(idxBase)[i]; break;
                        default:
                            idx = reinterpret_cast<const uint32_t*>(idxBase)[i]; break;
                    }
                    outIdx.push_back(indexOffset + idx);
                }
            } else {
                for (uint32_t i = 0; i < (uint32_t)vertCount; i++)
                    outIdx.push_back(indexOffset + i);
            }
        }
    }
}

// Extract float array from a GLTF accessor (handles packed/strided views)
static std::vector<float> extractFloats(const tinygltf::Model& m, int accIdx, int numComponents) {
    const auto& acc = m.accessors[accIdx];
    const auto& bv  = m.bufferViews[acc.bufferView];
    size_t stride   = bv.byteStride > 0 ? bv.byteStride : (size_t)(numComponents * sizeof(float));
    const uint8_t* base = m.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset;
    std::vector<float> out;
    out.reserve(acc.count * numComponents);
    for (size_t i = 0; i < acc.count; i++) {
        const float* f = reinterpret_cast<const float*>(base + i * stride);
        for (int c = 0; c < numComponents; c++) out.push_back(f[c]);
    }
    return out;
}

// Load a skinned GLTF mesh: fills outVerts, outIdx, and skinOut.
// Only processes model.skins[0] and the first mesh.
static bool loadSkinnedMeshData(const std::string& path,
                                std::vector<SkinnedVertex>& outVerts,
                                std::vector<uint32_t>& outIdx,
                                SkinData& skinOut)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;

    bool ok = (path.size() >= 4 && path.substr(path.size() - 4) == ".glb")
        ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
        : loader.LoadASCIIFromFile(&model, &err, &warn, path);

    if (!ok || model.skins.empty()) return false;

    const auto& skin = model.skins[0];
    const int numJoints = (int)skin.joints.size();
    skinOut.numJoints = numJoints;
    skinOut.inverseBindMatrices.resize(numJoints, glm::mat4(1.0f));
    skinOut.parentJoint.resize(numJoints, -1);
    skinOut.localBindPoses.resize(numJoints, glm::mat4(1.0f));

    // IBMs
    if (skin.inverseBindMatrices >= 0) {
        const auto& acc = model.accessors[skin.inverseBindMatrices];
        const auto& bv  = model.bufferViews[acc.bufferView];
        const float* data = reinterpret_cast<const float*>(
            model.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset);
        for (int i = 0; i < numJoints && i < (int)acc.count; i++) {
            // GLM matrices are column-major, GLTF IBMs are also column-major
            skinOut.inverseBindMatrices[i] = glm::make_mat4(data + i * 16);
        }
    }

    // Build joint index map: GLTF node index → skinData joint index
    std::map<int, int> nodeToJoint;
    for (int j = 0; j < numJoints; j++) nodeToJoint[skin.joints[j]] = j;

    // Parent chain: for each node, any node whose children[] contains it is its parent
    for (int ni = 0; ni < (int)model.nodes.size(); ni++) {
        for (int child : model.nodes[ni].children) {
            auto jIt = nodeToJoint.find(child);
            auto pIt = nodeToJoint.find(ni);
            if (jIt != nodeToJoint.end() && pIt != nodeToJoint.end())
                skinOut.parentJoint[jIt->second] = pIt->second;
        }
    }

    // Local bind poses from node TRS or matrix
    for (int j = 0; j < numJoints; j++) {
        const auto& node = model.nodes[skin.joints[j]];
        if (!node.matrix.empty()) {
            skinOut.localBindPoses[j] = glm::make_mat4(node.matrix.data());
        } else {
            glm::mat4 T(1), R(1), S(1);
            if (!node.translation.empty())
                T = glm::translate(glm::mat4(1.0f), glm::vec3((float)node.translation[0], (float)node.translation[1], (float)node.translation[2]));
            if (!node.rotation.empty()) {
                // GLTF quat: [x, y, z, w]; glm::quat ctor: (w, x, y, z)
                glm::quat q((float)node.rotation[3], (float)node.rotation[0], (float)node.rotation[1], (float)node.rotation[2]);
                R = glm::mat4_cast(q);
            }
            if (!node.scale.empty())
                S = glm::scale(glm::mat4(1.0f), glm::vec3((float)node.scale[0], (float)node.scale[1], (float)node.scale[2]));
            skinOut.localBindPoses[j] = T * R * S;
        }
    }

    // Parse animations
    for (const auto& anim : model.animations) {
        AnimationClip clip;
        clip.name = anim.name;

        // Map jointIndex → track
        std::map<int, JointTrack> trackMap;

        for (const auto& ch : anim.channels) {
            auto jIt = nodeToJoint.find(ch.target_node);
            if (jIt == nodeToJoint.end()) continue;
            int ji = jIt->second;

            const auto& sampler = anim.samplers[ch.sampler];
            if (sampler.interpolation != "LINEAR" && sampler.interpolation != "STEP") continue;

            auto times  = extractFloats(model, sampler.input, 1);
            auto values = extractFloats(model, sampler.output,
                (ch.target_path == "rotation") ? 4 : 3);

            if (clip.duration < times.back()) clip.duration = times.back();

            JointTrack& track = trackMap[ji];
            track.jointIndex = ji;

            if (ch.target_path == "translation") {
                for (size_t k = 0; k < times.size(); k++)
                    track.translations.push_back({times[k], {values[k*3], values[k*3+1], values[k*3+2]}});
            } else if (ch.target_path == "rotation") {
                for (size_t k = 0; k < times.size(); k++)
                    // GLTF [x,y,z,w] → glm::quat(w,x,y,z)
                    track.rotations.push_back({times[k], glm::quat(values[k*4+3], values[k*4], values[k*4+1], values[k*4+2])});
            } else if (ch.target_path == "scale") {
                for (size_t k = 0; k < times.size(); k++)
                    track.scales.push_back({times[k], {values[k*3], values[k*3+1], values[k*3+2]}});
            }
        }

        for (auto& [ji, tr] : trackMap) clip.tracks.push_back(std::move(tr));
        skinOut.animations.push_back(std::move(clip));
    }
    if (skinOut.animations.empty()) {
        // Add a dummy T-pose "animation"
        skinOut.animations.push_back({"T-Pose", 0.0f, {}});
    }

    // Extract skinned mesh vertices (first mesh, all primitives)
    for (const auto& mesh : model.meshes) {
        for (const auto& prim : mesh.primitives) {
            if (prim.mode != TINYGLTF_MODE_TRIANGLES) continue;

            auto fetchBase = [&](int accIdx, size_t& stride, size_t elemSize) -> const uint8_t* {
                const auto& acc = model.accessors[accIdx];
                const auto& bv  = model.bufferViews[acc.bufferView];
                stride = bv.byteStride > 0 ? bv.byteStride : elemSize;
                return model.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset;
            };

            auto posIt = prim.attributes.find("POSITION");
            if (posIt == prim.attributes.end()) continue;
            size_t posStride;
            const uint8_t* posBase = fetchBase(posIt->second, posStride, 12);
            const size_t vertCount = model.accessors[posIt->second].count;

            size_t normStride = 0; const uint8_t* normBase = nullptr;
            auto normIt = prim.attributes.find("NORMAL");
            if (normIt != prim.attributes.end()) normBase = fetchBase(normIt->second, normStride, 12);

            size_t uvStride = 0; const uint8_t* uvBase = nullptr;
            auto uvIt = prim.attributes.find("TEXCOORD_0");
            if (uvIt != prim.attributes.end()) uvBase = fetchBase(uvIt->second, uvStride, 8);

            // JOINTS_0 and WEIGHTS_0 accessors
            size_t jStride = 0; const uint8_t* jBase = nullptr;
            int    jCompType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
            auto jIt2 = prim.attributes.find("JOINTS_0");
            if (jIt2 != prim.attributes.end()) {
                const auto& acc = model.accessors[jIt2->second];
                jCompType = acc.componentType;
                const auto& bv = model.bufferViews[acc.bufferView];
                size_t elemSz = (jCompType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) ? 4 : 8;
                jStride = bv.byteStride > 0 ? bv.byteStride : elemSz;
                jBase   = model.buffers[bv.buffer].data.data() + bv.byteOffset + acc.byteOffset;
            }

            size_t wStride = 0; const uint8_t* wBase = nullptr;
            auto wIt = prim.attributes.find("WEIGHTS_0");
            if (wIt != prim.attributes.end()) wBase = fetchBase(wIt->second, wStride, 16);

            uint32_t indexOffset = (uint32_t)outVerts.size();

            for (size_t i = 0; i < vertCount; i++) {
                SkinnedVertex v{};
                const float* p = reinterpret_cast<const float*>(posBase + i * posStride);
                v.pos = {p[0], p[1], p[2]};
                if (normBase) {
                    const float* n = reinterpret_cast<const float*>(normBase + i * normStride);
                    v.normal = {n[0], n[1], n[2]};
                }
                if (uvBase) {
                    const float* uv = reinterpret_cast<const float*>(uvBase + i * uvStride);
                    v.texCoord = {uv[0], uv[1]};
                }
                if (jBase) {
                    if (jCompType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
                        const uint8_t* j = jBase + i * jStride;
                        v.joints = {j[0], j[1], j[2], j[3]};
                    } else {
                        const uint16_t* j = reinterpret_cast<const uint16_t*>(jBase + i * jStride);
                        v.joints = {j[0], j[1], j[2], j[3]};
                    }
                }
                if (wBase) {
                    const float* w = reinterpret_cast<const float*>(wBase + i * wStride);
                    v.weights = {w[0], w[1], w[2], w[3]};
                } else {
                    v.weights = {1.0f, 0.0f, 0.0f, 0.0f};
                }
                outVerts.push_back(v);
            }

            // Indices
            if (prim.indices >= 0) {
                const auto& idxAcc = model.accessors[prim.indices];
                const auto& idxBV  = model.bufferViews[idxAcc.bufferView];
                const uint8_t* idxBase = model.buffers[idxBV.buffer].data.data()
                                       + idxBV.byteOffset + idxAcc.byteOffset;
                for (size_t i = 0; i < idxAcc.count; i++) {
                    uint32_t idx = 0;
                    switch (idxAcc.componentType) {
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  idx = idxBase[i]; break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: idx = reinterpret_cast<const uint16_t*>(idxBase)[i]; break;
                        default: idx = reinterpret_cast<const uint32_t*>(idxBase)[i]; break;
                    }
                    outIdx.push_back(indexOffset + idx);
                }
            } else {
                for (uint32_t i = 0; i < (uint32_t)vertCount; i++)
                    outIdx.push_back(indexOffset + i);
            }
        }
    }
    return true;
}

void Renderer::loadSkinnedMesh(const std::string& modelName) {
    if (loadedSkinnedMeshes.count(modelName)) return;

    std::string path = workspace->activeProject.getModelsPath() + "/" + modelName;
    std::vector<SkinnedVertex> verts;
    std::vector<uint32_t>      indices;
    SkinData skin;

    if (!loadSkinnedMeshData(path, verts, indices, skin) || verts.empty()) {
        LOG_INFO("loadSkinnedMesh: no skin data in %s", modelName.c_str());
        return;
    }

    loadedSkins[modelName] = std::move(skin);

    SkinnedMeshResource res{};
    res.vertexCount = (uint32_t)verts.size();
    res.indexCount  = (uint32_t)indices.size();

    VkDeviceSize vSize = sizeof(SkinnedVertex) * verts.size();
    VkBuffer vStage; VkDeviceMemory vStageMem;
    createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStage, vStageMem);
    void* vd; vkMapMemory(device, vStageMem, 0, vSize, 0, &vd);
    memcpy(vd, verts.data(), vSize); vkUnmapMemory(device, vStageMem);
    createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, res.vertexBuffer, res.vertexBufferMemory);
    copyBuffer(vStage, res.vertexBuffer, vSize);
    vkDestroyBuffer(device, vStage, nullptr); vkFreeMemory(device, vStageMem, nullptr);

    VkDeviceSize iSize = sizeof(uint32_t) * indices.size();
    VkBuffer iStage; VkDeviceMemory iStageMem;
    createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, iStage, iStageMem);
    void* id_; vkMapMemory(device, iStageMem, 0, iSize, 0, &id_);
    memcpy(id_, indices.data(), iSize); vkUnmapMemory(device, iStageMem);
    createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, res.indexBuffer, res.indexBufferMemory);
    copyBuffer(iStage, res.indexBuffer, iSize);
    vkDestroyBuffer(device, iStage, nullptr); vkFreeMemory(device, iStageMem, nullptr);

    loadedSkinnedMeshes[modelName] = std::move(res);
    LOG_INFO("Skinned mesh loaded: %s (%u verts, %u idx, %d joints, %zu anims)",
        modelName.c_str(), res.vertexCount, res.indexCount,
        loadedSkins[modelName].numJoints, loadedSkins[modelName].animations.size());
}

void Renderer::loadMesh(const std::string& modelName) {
    if (loadedMeshes.count(modelName)) return;

    std::string path = workspace->activeProject.getModelsPath() + "/" + modelName;
    std::string ext  = std::filesystem::path(modelName).extension().string();

    std::vector<Vertex>   localVertices;
    std::vector<uint32_t> localIndices;

    try {
        if (ext == ".gltf" || ext == ".glb")
            loadMeshDataFromGLTF(path, localVertices, localIndices);
        else
            loadMeshDataFromOBJ(path, localVertices, localIndices);
    } catch (const std::exception& e) {
        LOG_WARNING("Mesh load failed (%s): %s — using fallback cube", modelName.c_str(), e.what());
    } catch (...) {
        LOG_WARNING("Mesh load failed (%s) — using fallback cube", modelName.c_str());
    }

    if (localVertices.empty() || localIndices.empty()) {
        if (loadedMeshes.count("__fallback"))
            loadedMeshes[modelName] = loadedMeshes["__fallback"];
        return;
    }

    MeshResource newMesh{};
    newMesh.indexCount  = (uint32_t)localIndices.size();
    newMesh.vertexCount = (uint32_t)localVertices.size();
    newMesh.cpuVerts    = localVertices;
    newMesh.cpuIdx      = localIndices;

    bool needsRT = rayTracer.isReady();
    VkBufferUsageFlags vbUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VkBufferUsageFlags ibUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (needsRT) {
        vbUsage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        ibUsage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    auto stagingUpload = [&](const void* data, VkDeviceSize size, VkBufferUsageFlags usage,
                              VkBuffer& outBuf, VkDeviceMemory& outMem) {
        VkBuffer stage; VkDeviceMemory stageMem;
        createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage, stageMem);
        void* p; vkMapMemory(device, stageMem, 0, size, 0, &p);
        memcpy(p, data, size); vkUnmapMemory(device, stageMem);
        createBuffer(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem, needsRT);
        copyBuffer(stage, outBuf, size);
        vkDestroyBuffer(device, stage, nullptr); vkFreeMemory(device, stageMem, nullptr);
    };

    stagingUpload(localVertices.data(), sizeof(Vertex)*localVertices.size(),
                  vbUsage, newMesh.vertexBuffer, newMesh.vertexBufferMemory);
    stagingUpload(localIndices.data(), sizeof(uint32_t)*localIndices.size(),
                  ibUsage, newMesh.indexBuffer, newMesh.indexBufferMemory);

    // Generate and upload LODs (skip for terrain meshes which are procedural)
    if (!modelName.starts_with("__terrain_") && localVertices.size() >= 12) {
        auto cb = [this](VkDeviceSize sz, VkBufferUsageFlags u, VkMemoryPropertyFlags p,
                         VkBuffer& b, VkDeviceMemory& m, bool rt) { createBuffer(sz,u,p,b,m,rt); };
        auto cpb = [this](VkBuffer s, VkBuffer d, VkDeviceSize sz) { copyBuffer(s,d,sz); };
        auto lod1 = simplifyMeshVC(localVertices, localIndices, 0.5f);
        auto lod2 = simplifyMeshVC(localVertices, localIndices, 0.25f);
        if (!lod1.idx.empty())
            uploadLODBuffers(device, cb, cpb, lod1.verts, lod1.idx,
                newMesh.lodVB[0], newMesh.lodVBMem[0], newMesh.lodIB[0], newMesh.lodIBMem[0], newMesh.lodIndexCount[0]);
        if (!lod2.idx.empty())
            uploadLODBuffers(device, cb, cpb, lod2.verts, lod2.idx,
                newMesh.lodVB[1], newMesh.lodVBMem[1], newMesh.lodIB[1], newMesh.lodIBMem[1], newMesh.lodIndexCount[1]);
    }

    const MeshResource& stored = loadedMeshes[modelName] = std::move(newMesh);
    LOG_INFO("Mesh incarcat: %s (%u verts, %u idx, LOD1=%u LOD2=%u)",
        modelName.c_str(), stored.vertexCount, stored.indexCount,
        stored.lodIndexCount[0], stored.lodIndexCount[1]);
}

void Renderer::createFallbackMesh() {
    if (loadedMeshes.count("__fallback")) return;

    std::vector<Vertex>   verts;
    std::vector<uint32_t> idx;

    auto face = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d, glm::vec3 n) {
        uint32_t base = (uint32_t)verts.size();
        glm::vec3 wh = {1,1,1};
        verts.push_back({a, wh, n, {0,0}});
        verts.push_back({b, wh, n, {1,0}});
        verts.push_back({c, wh, n, {1,1}});
        verts.push_back({d, wh, n, {0,1}});
        idx.insert(idx.end(), {base,base+1,base+2, base,base+2,base+3});
    };
    face({ .5f, .5f,-.5f},{ .5f, .5f, .5f},{ .5f,-.5f, .5f},{ .5f,-.5f,-.5f},{ 1, 0, 0});
    face({-.5f, .5f, .5f},{-.5f, .5f,-.5f},{-.5f,-.5f,-.5f},{-.5f,-.5f, .5f},{-1, 0, 0});
    face({-.5f, .5f, .5f},{ .5f, .5f, .5f},{ .5f, .5f,-.5f},{-.5f, .5f,-.5f},{ 0, 1, 0});
    face({-.5f,-.5f,-.5f},{ .5f,-.5f,-.5f},{ .5f,-.5f, .5f},{-.5f,-.5f, .5f},{ 0,-1, 0});
    face({-.5f, .5f, .5f},{-.5f,-.5f, .5f},{ .5f,-.5f, .5f},{ .5f, .5f, .5f},{ 0, 0, 1});
    face({ .5f, .5f,-.5f},{ .5f,-.5f,-.5f},{-.5f,-.5f,-.5f},{-.5f, .5f,-.5f},{ 0, 0,-1});

    MeshResource m{};
    m.indexCount  = (uint32_t)idx.size();
    m.vertexCount = (uint32_t)verts.size();
    m.cpuVerts    = verts;
    m.cpuIdx      = idx;

    auto upload = [&](const void* data, VkDeviceSize sz, VkBufferUsageFlags usage,
                      VkBuffer& outBuf, VkDeviceMemory& outMem) {
        VkBuffer stage; VkDeviceMemory stageMem;
        createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stage, stageMem);
        void* p; vkMapMemory(device, stageMem, 0, sz, 0, &p);
        memcpy(p, data, sz); vkUnmapMemory(device, stageMem);
        createBuffer(sz, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem);
        copyBuffer(stage, outBuf, sz);
        vkDestroyBuffer(device, stage, nullptr); vkFreeMemory(device, stageMem, nullptr);
    };
    upload(verts.data(), sizeof(Vertex)*verts.size(),
           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
           m.vertexBuffer, m.vertexBufferMemory);
    upload(idx.data(), sizeof(uint32_t)*idx.size(),
           VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
           m.indexBuffer, m.indexBufferMemory);

    loadedMeshes["__fallback"] = std::move(m);
}

void Renderer::loadAllProjectMeshes() {
    scanModelsFolder();
    for(const auto& model : availableModels) {
        loadMesh(model);
    }
}

// ---------------------------------------------------------------------------
// Pack all project assets into a single .rpak file (editor-only operation)
// ---------------------------------------------------------------------------
void Renderer::packAssets() {
    // Ensure all models are loaded so we have CPU data
    for (const auto& modelName : availableModels)
        loadMesh(modelName);

    std::string outputPath = workspace->activeProject.rootPath + "/" +
                             workspace->activeProject.name + ".rpak";

    struct BlobAsset { RpakEntry hdr; std::vector<uint8_t> data; };
    std::vector<BlobAsset> assets;

    auto appendBytes = [](std::vector<uint8_t>& blob, const void* p, size_t n) {
        blob.insert(blob.end(), (const uint8_t*)p, (const uint8_t*)p + n);
    };

    // --- Meshes ---
    for (const auto& modelName : availableModels) {
        auto it = loadedMeshes.find(modelName);
        if (it == loadedMeshes.end() || it->second.cpuVerts.empty()) continue;
        const MeshResource& mr = it->second;

        auto lod1 = simplifyMeshVC(mr.cpuVerts, mr.cpuIdx, 0.5f);
        auto lod2 = simplifyMeshVC(mr.cpuVerts, mr.cpuIdx, 0.25f);
        const SimpleMeshData* lods[3] = {
            nullptr, &lod1, &lod2
        };

        std::vector<uint8_t> blob;
        uint32_t lodCount = 3;
        appendBytes(blob, &lodCount, 4);
        for (int li = 0; li < 3; li++) {
            const std::vector<Vertex>*   vp = (li == 0) ? &mr.cpuVerts : &lods[li]->verts;
            const std::vector<uint32_t>* ip = (li == 0) ? &mr.cpuIdx   : &lods[li]->idx;
            uint32_t vc = (uint32_t)vp->size(), ic = (uint32_t)ip->size();
            appendBytes(blob, &vc, 4); appendBytes(blob, &ic, 4);
            appendBytes(blob, vp->data(), vc * sizeof(Vertex));
            appendBytes(blob, ip->data(), ic * sizeof(uint32_t));
        }

        RpakEntry entry{};
        entry.type = RpakAssetType::Mesh;
        strncpy(entry.name, modelName.c_str(), 254);
        assets.push_back({entry, std::move(blob)});
    }

    // --- Textures ---
    for (const auto& texName : availableTextures) {
        std::string path = workspace->activeProject.getTexturesPath() + "/" + texName;
        int w, h, ch;
        uint8_t* px = stbi_load(path.c_str(), &w, &h, &ch, 4);
        if (!px) continue;

        std::vector<uint8_t> blob;
        uint32_t uw = (uint32_t)w, uh = (uint32_t)h;
        appendBytes(blob, &uw, 4); appendBytes(blob, &uh, 4);
        appendBytes(blob, px, w * h * 4);
        stbi_image_free(px);

        RpakEntry entry{};
        entry.type = RpakAssetType::Texture;
        strncpy(entry.name, texName.c_str(), 254);
        assets.push_back({entry, std::move(blob)});
    }

    if (assets.empty()) { LOG_INFO("Nu exista assets de impachetat."); return; }

    // Compute blob offsets
    uint64_t dataStart = sizeof(RpakHeader) + sizeof(RpakEntry) * assets.size();
    uint64_t off = dataStart;
    for (auto& a : assets) { a.hdr.offset = off; a.hdr.size = a.data.size(); off += a.data.size(); }

    // Write file
    std::ofstream f(outputPath, std::ios::binary | std::ios::trunc);
    if (!f) { LOG_INFO("Eroare: Nu pot crea %s", outputPath.c_str()); return; }

    RpakHeader hdr{};
    hdr.count = (uint32_t)assets.size();
    f.write((char*)&hdr, sizeof(hdr));
    for (const auto& a : assets)  f.write((char*)&a.hdr,       sizeof(a.hdr));
    for (const auto& a : assets)  f.write((char*)a.data.data(), (std::streamsize)a.data.size());

    LOG_INFO("Ambalat %zu assets in %s (%.1f MB)",
        assets.size(), outputPath.c_str(), off / 1048576.0);
}

void Renderer::reloadCurrentModel() {
    if (availableModels.empty()) return;
    vkDeviceWaitIdle(device);
    loadMesh(availableModels[selectedModelIndex]);
}

void Renderer::scanAudioFolder() {
    availableAudio.clear();
    std::string path = workspace->activeProject.getAudioPath();
    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            auto ext = entry.path().extension();
            if (ext == ".wav" || ext == ".mp3" || ext == ".ogg" || ext == ".flac") {
                availableAudio.push_back(entry.path().filename().string());
            }
        }
    }
}

void Renderer::scanTexturesFolder() {
    availableTextures.clear();
    std::string path = workspace->activeProject.getTexturesPath();

    if (std::filesystem::exists(path) && std::filesystem::is_directory(path)) {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                availableTextures.push_back(entry.path().filename().string());
            }
        }
    }
}

void Renderer::createTextureImage(const std::string& path) {
    int texWidth, texHeight, texChannels;
    stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
        THROW_ENGINE_ERROR("Eroare fatala: Nu am putut incarca textura: " + path);
    }

    uint32_t mipLevels = static_cast<uint32_t>(
        std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(device, stagingBufferMemory);
    stbi_image_free(pixels);

    VkImage newImage;
    VkDeviceMemory newMemory;

    createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, newImage, newMemory, mipLevels);

    transitionImageLayout(newImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuffer, newImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
    generateMipmaps(newImage, texWidth, texHeight, mipLevels);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    textureImages.push_back(newImage);
    textureImageMemories.push_back(newMemory);
    textureMipLevels.push_back(mipLevels);
}

void Renderer::createTextureImageView() {
    VkImage img = textureImages.back();
    VkImageView newView;

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = img;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = textureMipLevels.back();
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    vkCreateImageView(device, &viewInfo, nullptr, &newView);

    textureImageViews.push_back(newView);
}

void Renderer::createTextureSampler() {
    VkSampler newSampler;
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(textureMipLevels.back());
    vkCreateSampler(device, &samplerInfo, nullptr, &newSampler);

    textureSamplers.push_back(newSampler);
}

void Renderer::reloadProjectTextures() {
    if (availableTextures.empty()) return; // no textures in new project — keep current ones

    vkDeviceWaitIdle(device);
    for (size_t i = 0; i < textureImages.size(); i++) {
        vkDestroyImageView(device, textureImageViews[i], nullptr);
        vkDestroyImage(device, textureImages[i], nullptr);
        vkFreeMemory(device, textureImageMemories[i], nullptr);
        vkDestroySampler(device, textureSamplers[i], nullptr);
    }
    textureImages.clear();
    textureImageViews.clear();
    textureImageMemories.clear();
    textureSamplers.clear();
    textureMipLevels.clear();

    for (const auto& texName : availableTextures) {
        std::string path = workspace->activeProject.getTexturesPath() + "/" + texName;
        try {
            createTextureImage(path);
            createTextureImageView();
            createTextureSampler();
        } catch (...) {
            LOG_INFO("Avertisment: Nu s-a putut incarca textura: %s", texName.c_str());
        }
    }
    if (!textureImages.empty())
        updateTextureDescriptorSet();
}

void Renderer::reloadCurrentTexture(int selectedEntityIndex) {
    if (selectedEntityIndex >= 0 && selectedEntityIndex < (int)scene->entities.size())
        scene->entities[selectedEntityIndex].textureIndex = selectedTextureIndex;
}

void Renderer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory, bool deviceAddress) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare: Nu s-a putut crea un buffer Vulkan!");
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, buffer, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
    if (deviceAddress) allocInfo.pNext = &allocFlags;

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare: Nu s-a putut aloca memoria video pentru buffer!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void Renderer::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void Renderer::createGridMesh() {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    int size = 30;
    float t = 0.008f;

    for (int i = -size; i <= size; i++) {
        glm::vec3 cX = (i == 0) ? glm::vec3(1.0f, 0.2f, 0.2f) : glm::vec3(0.3f);
        uint32_t bX = vertices.size();
        vertices.push_back({{ -size, (float)i - t, -0.01f }, cX, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        vertices.push_back({{  size, (float)i - t, -0.01f }, cX, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        vertices.push_back({{ -size, (float)i + t, -0.01f }, cX, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        vertices.push_back({{  size, (float)i + t, -0.01f }, cX, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        indices.push_back(bX); indices.push_back(bX+1); indices.push_back(bX+2);
        indices.push_back(bX+2); indices.push_back(bX+1); indices.push_back(bX+3);

        glm::vec3 cY = (i == 0) ? glm::vec3(0.2f, 1.0f, 0.2f) : glm::vec3(0.3f);
        uint32_t bY = vertices.size();
        vertices.push_back({{ (float)i - t, -size, -0.01f }, cY, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        vertices.push_back({{ (float)i + t, -size, -0.01f }, cY, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        vertices.push_back({{ (float)i - t,  size, -0.01f }, cY, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        vertices.push_back({{ (float)i + t,  size, -0.01f }, cY, {0.0f, 0.0f, 1.0f}, {0.5f, 0.5f}});
        indices.push_back(bY); indices.push_back(bY+1); indices.push_back(bY+2);
        indices.push_back(bY+2); indices.push_back(bY+1); indices.push_back(bY+3);
    }

    gridMesh.indexCount = static_cast<uint32_t>(indices.size());

    VkDeviceSize vSize = sizeof(Vertex) * vertices.size();
    VkBuffer vStagingBuffer;
    VkDeviceMemory vStagingMemory;
    createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vStagingBuffer, vStagingMemory);
    void* vData; vkMapMemory(device, vStagingMemory, 0, vSize, 0, &vData);
    memcpy(vData, vertices.data(), (size_t)vSize); vkUnmapMemory(device, vStagingMemory);
    createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gridMesh.vertexBuffer, gridMesh.vertexBufferMemory);
    copyBuffer(vStagingBuffer, gridMesh.vertexBuffer, vSize);
    vkDestroyBuffer(device, vStagingBuffer, nullptr); vkFreeMemory(device, vStagingMemory, nullptr);

    VkDeviceSize iSize = sizeof(uint32_t) * indices.size();
    VkBuffer iStagingBuffer;
    VkDeviceMemory iStagingMemory;
    createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, iStagingBuffer, iStagingMemory);
    void* iData; vkMapMemory(device, iStagingMemory, 0, iSize, 0, &iData);
    memcpy(iData, indices.data(), (size_t)iSize); vkUnmapMemory(device, iStagingMemory);
    createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gridMesh.indexBuffer, gridMesh.indexBufferMemory);
    copyBuffer(iStagingBuffer, gridMesh.indexBuffer, iSize);
    vkDestroyBuffer(device, iStagingBuffer, nullptr); vkFreeMemory(device, iStagingMemory, nullptr);
}

// --- Terrain Mesh Generation ---

// Portable hash giving a pseudo-random float in [0, 1) for grid coords + seed
static float terrainHash(int x, int y, int seed) {
    unsigned int n = (unsigned int)(x * 1619 + y * 31337 + seed * 6971);
    n = (n << 13) ^ n;
    return ((n * (n * n * 15731u + 789221u) + 1376312589u) & 0x7fffffffu) / 2147483648.0f;
}

// Bilinear-interpolated value noise
static float terrainSmooth(float x, float y, int seed) {
    int ix = (int)glm::floor(x), iy = (int)glm::floor(y);
    float fx = x - ix, fy = y - iy;
    float ux = fx * fx * (3.0f - 2.0f * fx); // smoothstep
    float uy = fy * fy * (3.0f - 2.0f * fy);
    return glm::mix(
        glm::mix(terrainHash(ix,   iy,   seed), terrainHash(ix+1, iy,   seed), ux),
        glm::mix(terrainHash(ix,   iy+1, seed), terrainHash(ix+1, iy+1, seed), ux),
        uy);
}

// FBM: 6 octaves, returns value in approximately [0, amplitude]
static float terrainFBM(float x, float y, float frequency, float amplitude, int seed) {
    float value = 0.0f, f = frequency, a = amplitude;
    for (int i = 0; i < 6; i++) {
        value += a * terrainSmooth(x * f, y * f, seed + i * 97);
        f *= 2.0f; a *= 0.5f;
    }
    return value;
}

void Renderer::destroyMesh(const std::string& key) {
    auto it = loadedMeshes.find(key);
    if (it == loadedMeshes.end()) return;
    vkDeviceWaitIdle(device);
    MeshResource& m = it->second;
    vkDestroyBuffer(device, m.vertexBuffer,      nullptr);
    vkFreeMemory(   device, m.vertexBufferMemory, nullptr);
    vkDestroyBuffer(device, m.indexBuffer,        nullptr);
    vkFreeMemory(   device, m.indexBufferMemory,  nullptr);
    for (int i = 0; i < 2; i++) {
        if (m.lodVB[i] != VK_NULL_HANDLE) { vkDestroyBuffer(device, m.lodVB[i], nullptr); vkFreeMemory(device, m.lodVBMem[i], nullptr); }
        if (m.lodIB[i] != VK_NULL_HANDLE) { vkDestroyBuffer(device, m.lodIB[i], nullptr); vkFreeMemory(device, m.lodIBMem[i], nullptr); }
    }
    loadedMeshes.erase(it);
}

void Renderer::generateTerrainMesh(const GameObject& entity) {
    const std::string key = "__terrain_" + std::to_string(entity.id);
    destroyMesh(key);  // free any previous version

    int W = glm::max(entity.terrainWidth,  2);
    int D = glm::max(entity.terrainDepth,  2);
    float freq = entity.terrainFrequency;
    float amp  = entity.terrainAmplitude;
    int   seed = entity.terrainSeed;

    // Grid of (W+1)*(D+1) vertices; quads span from (-W/2, -D/2) to (W/2, D/2) in XY
    auto H = [&](int i, int j) {
        return terrainFBM((float)i, (float)j, freq, amp, seed);
    };

    std::vector<Vertex>   verts;
    std::vector<uint32_t> idxs;
    verts.reserve((W+1) * (D+1));
    idxs.reserve(W * D * 6);

    for (int j = 0; j <= D; j++) {
        for (int i = 0; i <= W; i++) {
            float x = (float)i - W * 0.5f;
            float y = (float)j - D * 0.5f;
            float z = H(i, j);

            // Finite-difference normal (forward diff at edges, central elsewhere)
            float hL = H(glm::max(i-1, 0), j);
            float hR = H(glm::min(i+1, W), j);
            float hD = H(i, glm::max(j-1, 0));
            float hU = H(i, glm::min(j+1, D));
            float dzdx = (hR - hL) / ((i > 0 && i < W) ? 2.0f : 1.0f);
            float dzdy = (hU - hD) / ((j > 0 && j < D) ? 2.0f : 1.0f);
            glm::vec3 normal = glm::normalize(glm::vec3(-dzdx, -dzdy, 1.0f));

            Vertex v{};
            v.pos      = { x, y, z };
            v.color    = { 1.0f, 1.0f, 1.0f };
            v.normal   = normal;
            v.texCoord = { (float)i / (float)W * 4.0f, (float)j / (float)D * 4.0f }; // tiled 4×
            verts.push_back(v);
        }
    }

    for (int j = 0; j < D; j++) {
        for (int i = 0; i < W; i++) {
            uint32_t tl = j * (W+1) + i;
            uint32_t tr = tl + 1;
            uint32_t bl = tl + (W+1);
            uint32_t br = bl + 1;
            idxs.push_back(tl); idxs.push_back(bl); idxs.push_back(tr);
            idxs.push_back(tr); idxs.push_back(bl); idxs.push_back(br);
        }
    }

    MeshResource m{};
    m.vertexCount = (uint32_t)verts.size();
    m.indexCount  = (uint32_t)idxs.size();

    VkDeviceSize vSize = sizeof(Vertex) * verts.size();
    VkBuffer vStage; VkDeviceMemory vStageMem;
    createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vStage, vStageMem);
    void* vd; vkMapMemory(device, vStageMem, 0, vSize, 0, &vd);
    memcpy(vd, verts.data(), vSize); vkUnmapMemory(device, vStageMem);
    createBuffer(vSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.vertexBuffer, m.vertexBufferMemory);
    copyBuffer(vStage, m.vertexBuffer, vSize);
    vkDestroyBuffer(device, vStage, nullptr); vkFreeMemory(device, vStageMem, nullptr);

    VkDeviceSize iSize = sizeof(uint32_t) * idxs.size();
    VkBuffer iStage; VkDeviceMemory iStageMem;
    createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 iStage, iStageMem);
    void* id_; vkMapMemory(device, iStageMem, 0, iSize, 0, &id_);
    memcpy(id_, idxs.data(), iSize); vkUnmapMemory(device, iStageMem);
    createBuffer(iSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m.indexBuffer, m.indexBufferMemory);
    copyBuffer(iStage, m.indexBuffer, iSize);
    vkDestroyBuffer(device, iStage, nullptr); vkFreeMemory(device, iStageMem, nullptr);

    loadedMeshes[key] = m;
    LOG_INFO("Teren generat: %s (%dx%d, %u verts)", entity.name.c_str(), W, D, m.vertexCount);

    // Keep RT BLAS in sync with the newly (re)generated terrain mesh
    if (rayTracer.isReady())
        rayTracer.buildBLAS(key, loadedMeshes[key].vertexBuffer, loadedMeshes[key].vertexCount,
                            loadedMeshes[key].indexBuffer, loadedMeshes[key].indexCount,
                            commandPool, graphicsQueue);
}
