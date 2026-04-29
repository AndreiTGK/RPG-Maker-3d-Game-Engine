#include "Renderer.hpp"
#include "EngineLog.hpp"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"
#include "ImGuizmo.h"
#include <glm/gtc/type_ptr.hpp>
#include <queue>
#include <set>
#include <algorithm>

std::tuple<std::vector<RtInstance>, std::vector<std::string>, std::vector<std::pair<VkBuffer,VkBuffer>>>
Renderer::buildRtData() {
    // Ordered mesh list (alphabetical, same as std::map iteration order)
    std::vector<std::string> meshNames;
    std::vector<std::pair<VkBuffer,VkBuffer>> meshBuffers;
    for (auto& [name, mesh] : loadedMeshes) {
        if (mesh.vertexBuffer != VK_NULL_HANDLE && mesh.indexBuffer != VK_NULL_HANDLE) {
            meshNames.push_back(name);
            meshBuffers.push_back({mesh.vertexBuffer, mesh.indexBuffer});
        }
    }

    // Per-entity RT instances
    std::vector<RtInstance> instances;
    for (auto& e : scene->entities) {
        if (!e.visible) continue;
        std::string meshKey;
        if (e.isTerrain) {
            meshKey = "__terrain_" + std::to_string(e.id);
        } else if (!availableModels.empty() && e.modelIndex >= 0 && (size_t)e.modelIndex < availableModels.size()) {
            meshKey = availableModels[e.modelIndex];
        }
        if (meshKey.empty() || !loadedMeshes.count(meshKey)) {
            if (!loadedMeshes.count("__fallback")) continue;
            meshKey = "__fallback";
        }
        RtInstance ri;
        ri.transform = e.transform.mat4();
        ri.meshName = meshKey;
        ri.textureIndex = e.textureIndex >= 0 ? static_cast<uint32_t>(e.textureIndex) : 0;
        instances.push_back(ri);
    }

    return {instances, meshNames, meshBuffers};
}

void Renderer::updateUniformBuffer() {
    UniformBufferObject ubo{};
    ubo.view = camera->getViewMatrix();
    ubo.proj = camera->getProjMatrix(windowExtent.width / (float)windowExtent.height);

    ubo.ambientLight = scene->ambientLight;
    ubo.skyColor     = scene->skyColor;

    glm::vec3 sunDir = scene->sunDirection;
    if (glm::length(sunDir) < 0.001f) sunDir = glm::vec3(0.0f, -1.0f, 0.001f);
    else sunDir = glm::normalize(sunDir);
    ubo.sunDirection   = sunDir;
    ubo.shadowsEnabled = scene->shadowsEnabled ? 1.0f : 0.0f;

    // Pack up to 4 light-source entities into the UBO array using world-space positions
    ubo.numActiveLights = 0;
    for (int i = 0; i < (int)scene->entities.size(); i++) {
        const auto& entity = scene->entities[i];
        if (!entity.isLightSource) continue;
        if (ubo.numActiveLights >= 4) break;

        glm::mat4 worldMat = getWorldTransform(*scene, i);
        GpuPointLight& gl = ubo.lights[ubo.numActiveLights];
        gl.position  = glm::vec3(worldMat[3]);  // world-space position from last column
        gl.intensity = entity.lightIntensity;
        gl.color     = entity.lightColor;
        gl.radius    = entity.lightRadius;
        ubo.numActiveLights++;
    }

    // Light-space matrices: perspective from each active point light (tile i),
    // or ortho from sun into tile 0 when there are no active lights.
    glm::mat4 clipMatrix = glm::mat4(1.0f);
    clipMatrix[1][1] = -1.0f;
    clipMatrix[2][2] = 0.5f;
    clipMatrix[3][2] = 0.5f;

    // lightSpaceMatrix[0] is always the sun/directional shadow matrix.
    // Point lights use the cubemap shadow system; lightSpaceMatrix[1..3] are unused.
    {
        glm::vec3 safeUp = glm::vec3(0.0f, 0.0f, 1.0f);
        if (glm::abs(sunDir.z) > 0.99f) safeUp = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 shadowCameraPos = camera->pos - sunDir * 50.0f;
        glm::mat4 lightView = glm::lookAt(shadowCameraPos, camera->pos, safeUp);
        glm::mat4 lightProj = glm::ortho(-20.0f, 20.0f, -20.0f, 20.0f, 0.1f, 200.0f);
        ubo.lightSpaceMatrix[0]     = clipMatrix * lightProj * lightView;
        cachedLightSpaceMatrices[0] = ubo.lightSpaceMatrix[0];
        for (int i = 1; i < 4; i++) {
            ubo.lightSpaceMatrix[i]     = glm::mat4(1.0f);
            cachedLightSpaceMatrices[i] = glm::mat4(1.0f);
        }
    }

    memcpy(uniformBufferMapped, &ubo, sizeof(ubo));
}


// --- Frustum culling helpers ---

// A normalized frustum plane: dot(normal, point) + d >= 0 means inside.
struct FrustumPlane { glm::vec3 normal; float d; };

// Extract 6 frustum planes from a combined view-projection matrix.
// Uses the Gribb/Hartmann method adapted for Vulkan's [0,1] depth range
// (GLM_FORCE_DEPTH_ZERO_TO_ONE) so the near plane is row2, not row3+row2.
static void extractFrustumPlanes(const glm::mat4& vp, FrustumPlane planes[6]) {
    // GLM is column-major: vp[col][row]. Row i = (vp[0][i], vp[1][i], vp[2][i], vp[3][i]).
    auto row = [&](int i) {
        return glm::vec4(vp[0][i], vp[1][i], vp[2][i], vp[3][i]);
    };
    glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);

    auto toPlane = [](glm::vec4 p) -> FrustumPlane {
        float len = glm::length(glm::vec3(p));
        if (len < 1e-6f) return {{0,0,0}, 0};
        return { glm::vec3(p) / len, p.w / len };
    };

    planes[0] = toPlane(r3 + r0); // Left
    planes[1] = toPlane(r3 - r0); // Right
    planes[2] = toPlane(r3 + r1); // Bottom
    planes[3] = toPlane(r3 - r1); // Top
    planes[4] = toPlane(r2);      // Near  (depth >= 0 in [0,1] NDC)
    planes[5] = toPlane(r3 - r2); // Far   (depth <= 1 in [0,1] NDC)
}

// Returns false if the sphere is entirely outside any frustum half-space.
static bool sphereInFrustum(const FrustumPlane planes[6], glm::vec3 center, float radius) {
    for (int i = 0; i < 6; i++) {
        if (glm::dot(planes[i].normal, center) + planes[i].d < -radius)
            return false;
    }
    return true;
}

void Renderer::renderScene(VkCommandBuffer commandBuffer) {
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    SimplePushConstantData gridPush{};
    gridPush.modelMatrix = glm::mat4(1.0f);
    gridPush.textureIndex = kUnlitTextureIndex;
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SimplePushConstantData), &gridPush);

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &gridMesh.vertexBuffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, gridMesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(commandBuffer, gridMesh.indexCount, 1, 0, 0, 0);

    // Build frustum planes once per frame from the current camera VP matrix
    glm::mat4 view = camera->getViewMatrix();
    glm::mat4 proj = camera->getProjMatrix(windowExtent.width / (float)windowExtent.height);
    FrustumPlane frustum[6];
    extractFrustumPlanes(proj * view, frustum);

    for (int ei = 0; ei < (int)scene->entities.size(); ei++) {
        auto& entity = scene->entities[ei]; // non-const: terrain clears dirty flag

        if (!entity.visible) continue;

        // Determine which mesh to use for this entity
        std::string meshKey;
        if (entity.isTerrain) {
            meshKey = "__terrain_" + std::to_string(entity.id);
            if (entity.terrainDirty || !loadedMeshes.count(meshKey)) {
                generateTerrainMesh(entity);
                entity.terrainDirty = false;
            }
        } else {
            if (availableModels.empty() || entity.modelIndex < 0 ||
                (size_t)entity.modelIndex >= availableModels.size()) continue;
            meshKey = availableModels[entity.modelIndex];
        }

        if (!loadedMeshes.count(meshKey)) {
            if (!loadedMeshes.count("__fallback")) continue;
            meshKey = "__fallback";
        }
        MeshResource& mesh = loadedMeshes[meshKey];

        glm::mat4 worldMat = getWorldTransform(*scene, ei);
        glm::vec3 worldPos = glm::vec3(worldMat[3]);
        glm::vec3 worldScale = {
            glm::length(glm::vec3(worldMat[0])),
            glm::length(glm::vec3(worldMat[1])),
            glm::length(glm::vec3(worldMat[2]))
        };
        // Terrain bounding radius: use diagonal of the grid
        float radius = entity.isTerrain
            ? glm::length(glm::vec3(entity.terrainWidth, entity.terrainDepth, entity.terrainAmplitude) * 0.5f)
            : glm::max(worldScale.x, glm::max(worldScale.y, worldScale.z));

        if (!sphereInFrustum(frustum, worldPos, radius))
            continue;

        SimplePushConstantData pushData{};
        pushData.modelMatrix  = worldMat;
        int maxTex = textureImages.empty() ? 0 : (int)textureImages.size() - 1;
        pushData.textureIndex = std::clamp(entity.textureIndex, 0, maxTex);
        pushData.metallic     = entity.metallic;
        pushData.roughness    = entity.roughness;

        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SimplePushConstantData), &pushData);

        // LOD selection: normalise distance by bounding radius so large objects stay sharp longer
        float distToCamera   = glm::length(worldPos - camera->pos);
        float normalisedDist = distToCamera / (radius + 1.0f);
        if (normalisedDist > 12.0f && mesh.lodIB[1] != VK_NULL_HANDLE && mesh.lodIndexCount[1] > 0) {
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mesh.lodVB[1], offsets);
            vkCmdBindIndexBuffer(commandBuffer, mesh.lodIB[1], 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, mesh.lodIndexCount[1], 1, 0, 0, 0);
        } else if (normalisedDist > 4.0f && mesh.lodIB[0] != VK_NULL_HANDLE && mesh.lodIndexCount[0] > 0) {
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mesh.lodVB[0], offsets);
            vkCmdBindIndexBuffer(commandBuffer, mesh.lodIB[0], 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, mesh.lodIndexCount[0], 1, 0, 0, 0);
        } else {
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mesh.vertexBuffer, offsets);
            vkCmdBindIndexBuffer(commandBuffer, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, mesh.indexCount, 1, 0, 0, 0);
        }
    }
}


void Renderer::drawFrame(float deltaTime, bool isPlaying) {
    vkWaitForFences(device, 1, &inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &inFlightFence);

    // Tick particles and skinning here — after the fence wait — so any buffer destroy/recreate
    // triggered by parameter changes is safe (previous frame GPU work is done).
    tickParticles(deltaTime);
    tickSkinning(deltaTime);

    uint32_t imgIdx;
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imgIdx);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        THROW_ENGINE_ERROR("Eroare la achizitionarea imaginii SwapChain!");
    }

    updateUniformBuffer();

    if (useRayTracing && rayTracer.isReady()) {
        auto [rtInst, meshNames, meshBuffers] = buildRtData();
        rayTracer.buildTLAS(rtInst, meshNames, commandPool, graphicsQueue);
        rayTracer.createDescriptorSets(uniformBuffer, sizeof(UniformBufferObject), meshBuffers, textureImageViews, textureSamplers);
    }

    vkResetCommandBuffer(commandBuffer, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(commandBuffer, &bi);

    if (!useRayTracing) {
        // --- Pass 1a: Sun directional shadow into atlas tile 0 (2048×2048) ---
        {
            VkClearValue sunClear; sunClear.depthStencil = {1.0f, 0};
            VkRenderPassBeginInfo sunBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            sunBI.renderPass        = shadowRenderPass;
            sunBI.framebuffer       = shadowFramebuffer;
            sunBI.renderArea.extent = {4096, 4096};
            sunBI.clearValueCount   = 1;
            sunBI.pClearValues      = &sunClear;
            vkCmdBeginRenderPass(commandBuffer, &sunBI, VK_SUBPASS_CONTENTS_INLINE);

            if (scene->shadowsEnabled) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
                vkCmdSetDepthBias(commandBuffer, 1.25f, 0.0f, 1.75f);
                vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        shadowPipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
                VkViewport sunVP{0.0f, 0.0f, 2048.0f, 2048.0f, 0.0f, 1.0f};
                VkRect2D   sunSC{{0, 0}, {2048, 2048}};
                vkCmdSetViewport(commandBuffer, 0, 1, &sunVP);
                vkCmdSetScissor(commandBuffer, 0, 1, &sunSC);
                for (int si = 0; si < (int)scene->entities.size(); si++) {
                    const auto& entity = scene->entities[si];
                    if (entity.isLightSource) continue;
                    std::string meshKey = entity.isTerrain
                        ? "__terrain_" + std::to_string(entity.id)
                        : (!availableModels.empty() && entity.modelIndex >= 0 &&
                           (size_t)entity.modelIndex < availableModels.size()
                           ? availableModels[entity.modelIndex] : "");
                    if (meshKey.empty() || !loadedMeshes.count(meshKey)) continue;
                    MeshResource& mesh = loadedMeshes[meshKey];
                    ShadowPushConstant spush{getWorldTransform(*scene, si), cachedLightSpaceMatrices[0]};
                    vkCmdPushConstants(commandBuffer, shadowPipelineLayout,
                                       VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPushConstant), &spush);
                    VkDeviceSize off = 0;
                    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mesh.vertexBuffer, &off);
                    vkCmdBindIndexBuffer(commandBuffer, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                    vkCmdDrawIndexed(commandBuffer, mesh.indexCount, 1, 0, 0, 0);
                }
            }
            vkCmdEndRenderPass(commandBuffer);
        }

        // --- Pass 1b: Omnidirectional cubemap shadows for each active point light ---
        // Collect light data from scene (same logic as updateUniformBuffer)
        struct CubeLightInfo { glm::vec3 pos; float radius; };
        std::vector<CubeLightInfo> activeLights;
        for (int ei = 0; ei < (int)scene->entities.size() && (int)activeLights.size() < (int)kMaxCubeLights; ei++) {
            const auto& e = scene->entities[ei];
            if (!e.isLightSource) continue;
            glm::mat4 w = getWorldTransform(*scene, ei);
            activeLights.push_back({glm::vec3(w[3]), e.lightRadius});
        }
        if (scene->shadowsEnabled && !activeLights.empty()) {
            // 6 canonical face directions (Vulkan cube face convention)
            static const glm::vec3 faceDirs[6] = {
                { 1, 0, 0}, {-1, 0, 0},
                { 0, 1, 0}, { 0,-1, 0},
                { 0, 0, 1}, { 0, 0,-1}
            };
            static const glm::vec3 faceUps[6] = {
                {0, 0,-1}, {0, 0,-1},  // ±X: up = -Z
                {0, 0,-1}, {0, 0,-1},  // ±Y: up = -Z
                {0,-1, 0}, {0,-1, 0}   // ±Z: up = -Y
            };

            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowCubePipeline);
            VkViewport cubeVP{0.0f, 0.0f, float(kCubeShadowRes), float(kCubeShadowRes), 0.0f, 1.0f};
            VkRect2D   cubeSC{{0, 0}, {kCubeShadowRes, kCubeShadowRes}};
            vkCmdSetViewport(commandBuffer, 0, 1, &cubeVP);
            vkCmdSetScissor(commandBuffer, 0, 1, &cubeSC);

            // 90° perspective; Y-flip for Vulkan clip space
            glm::mat4 cubeProj = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 200.0f);
            cubeProj[1][1] *= -1.0f;

            for (int li = 0; li < (int)activeLights.size(); li++) {
                glm::vec3 lightPos    = activeLights[li].pos;
                float     lightRadius = activeLights[li].radius;
                glm::vec4 lightPosRadius{lightPos, lightRadius};

                for (int f = 0; f < 6; f++) {
                    uint32_t fbIdx = uint32_t(li) * 6 + uint32_t(f);

                    std::array<VkClearValue, 2> clearVals{};
                    clearVals[0].color        = {1.0f, 0.0f, 0.0f, 0.0f}; // max linear depth = no shadow
                    clearVals[1].depthStencil = {1.0f, 0};

                    VkRenderPassBeginInfo faceBI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                    faceBI.renderPass        = shadowCubeRenderPass;
                    faceBI.framebuffer       = shadowCubeFBs[fbIdx];
                    faceBI.renderArea.extent = {kCubeShadowRes, kCubeShadowRes};
                    faceBI.clearValueCount   = 2;
                    faceBI.pClearValues      = clearVals.data();
                    vkCmdBeginRenderPass(commandBuffer, &faceBI, VK_SUBPASS_CONTENTS_INLINE);

                    glm::mat4 faceView     = glm::lookAt(lightPos, lightPos + faceDirs[f], faceUps[f]);
                    glm::mat4 faceViewProj = cubeProj * faceView;

                    for (int si = 0; si < (int)scene->entities.size(); si++) {
                        const auto& entity = scene->entities[si];
                        if (entity.isLightSource) continue;
                        std::string meshKey = entity.isTerrain
                            ? "__terrain_" + std::to_string(entity.id)
                            : (!availableModels.empty() && entity.modelIndex >= 0 &&
                               (size_t)entity.modelIndex < availableModels.size()
                               ? availableModels[entity.modelIndex] : "");
                        if (meshKey.empty() || !loadedMeshes.count(meshKey)) continue;
                        MeshResource& mesh = loadedMeshes[meshKey];

                        CubeShadowPushConstant cp{};
                        cp.modelMatrix    = getWorldTransform(*scene, si);
                        cp.faceViewProj   = faceViewProj;
                        cp.lightPosRadius = lightPosRadius;
                        vkCmdPushConstants(commandBuffer, shadowCubePipelineLayout,
                                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                           0, sizeof(CubeShadowPushConstant), &cp);
                        VkDeviceSize off = 0;
                        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mesh.vertexBuffer, &off);
                        vkCmdBindIndexBuffer(commandBuffer, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                        vkCmdDrawIndexed(commandBuffer, mesh.indexCount, 1, 0, 0, 0);
                    }
                    vkCmdEndRenderPass(commandBuffer);
                }
            }
        }

        // --- Pass 2: HDR scene → hdrImage (R16G16B16A16_SFLOAT) ---
        std::array<VkClearValue, 2> hdrClear{};
        hdrClear[0].color = {{scene->skyColor.r, scene->skyColor.g, scene->skyColor.b, 1.0f}};
        hdrClear[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo hdrRPI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        hdrRPI.renderPass = renderPass;       // HDR scene render pass
        hdrRPI.framebuffer = hdrFramebuffer;  // single offscreen framebuffer
        hdrRPI.renderArea.extent = windowExtent;
        hdrRPI.clearValueCount = 2;
        hdrRPI.pClearValues = hdrClear.data();
        vkCmdBeginRenderPass(commandBuffer, &hdrRPI, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        VkViewport vp{};
        vp.x = 0.0f;
        vp.y = (float)windowExtent.height;
        vp.width = (float)windowExtent.width;
        vp.height = -(float)windowExtent.height;
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &vp);
        VkRect2D sc{{0, 0}, windowExtent};
        vkCmdSetScissor(commandBuffer, 0, 1, &sc);
        renderScene(commandBuffer);
        renderSkinned(commandBuffer);   // skinned meshes use depth-write, draw before particles
        renderParticles(commandBuffer); // alpha-blended on top of scene, before post pass
        vkCmdEndRenderPass(commandBuffer);

        // Barrier: hdrImage layout already transitioned to SHADER_READ_ONLY by render pass;
        // ensure color writes are visible before fragment shader reads in the post pass.
        VkImageMemoryBarrier hdrBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        hdrBarrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        hdrBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        hdrBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hdrBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        hdrBarrier.image = hdrImage;
        hdrBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        hdrBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        hdrBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &hdrBarrier);

        // --- Pass 3: Post-processing (ACES + bloom) + ImGui → swapchain ---
        VkClearValue postClear{};
        postClear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        VkRenderPassBeginInfo postRPI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        postRPI.renderPass = postRenderPass;
        postRPI.framebuffer = swapChainFramebuffers[imgIdx];
        postRPI.renderArea.extent = windowExtent;
        postRPI.clearValueCount = 1;
        postRPI.pClearValues = &postClear;
        vkCmdBeginRenderPass(commandBuffer, &postRPI, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &postDescriptorSet, 0, nullptr);
        VkViewport postVP{0.0f, 0.0f, (float)windowExtent.width, (float)windowExtent.height, 0.0f, 1.0f};
        vkCmdSetViewport(commandBuffer, 0, 1, &postVP);
        vkCmdSetScissor(commandBuffer, 0, 1, &sc);
        vkCmdPushConstants(commandBuffer, postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PostSettings), &postSettings);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0); // fullscreen triangle

        renderInGameUI(commandBuffer, isPlaying); // in-game UI overlay (play mode only)
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
        vkCmdEndRenderPass(commandBuffer);

    } else {
        // --- Ray tracing path ---
        if (rayTracer.isReady() && rayTracer.rtDescriptorSet != VK_NULL_HANDLE) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracer.rtPipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracer.rtPipelineLayout, 0, 1, &rayTracer.rtDescriptorSet, 0, nullptr);
            static auto pfn_vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
            pfn_vkCmdTraceRaysKHR(commandBuffer, &rayTracer.raygenRegion, &rayTracer.missRegion, &rayTracer.hitRegion, &rayTracer.callRegion, windowExtent.width, windowExtent.height, 1);

            VkImageMemoryBarrier storageBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            storageBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            storageBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            storageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            storageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            storageBarrier.image = rayTracer.storageImage;
            storageBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            storageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            storageBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            VkImageMemoryBarrier swapBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            swapBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            swapBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapBarrier.image = swapChainImages[imgIdx];
            swapBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            swapBarrier.srcAccessMask = 0;
            swapBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &storageBarrier);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapBarrier);

            VkImageBlit blit{};
            blit.srcOffsets[1] = {(int32_t)windowExtent.width, (int32_t)windowExtent.height, 1};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.dstOffsets[1] = {(int32_t)windowExtent.width, (int32_t)windowExtent.height, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            vkCmdBlitImage(commandBuffer, rayTracer.storageImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapChainImages[imgIdx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);

            storageBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            storageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            storageBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            storageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            swapBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapBarrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            swapBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            swapBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 0, nullptr, 0, nullptr, 1, &storageBarrier);
            vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapBarrier);
        }

        // If RT wasn't ready the swapchain image hasn't been transitioned; do it now.
        if (!(rayTracer.isReady() && rayTracer.rtDescriptorSet != VK_NULL_HANDLE)) {
            VkImageMemoryBarrier swapFallback{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            swapFallback.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            swapFallback.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            swapFallback.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapFallback.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapFallback.image = swapChainImages[imgIdx];
            swapFallback.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            swapFallback.srcAccessMask = 0;
            swapFallback.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            vkCmdPipelineBarrier(commandBuffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0, 0, nullptr, 0, nullptr, 1, &swapFallback);
        }

        // ImGui only pass over the RT result (LOAD_OP_LOAD, compatible with postRenderPass)
        VkRenderPassBeginInfo rtImguiRPI{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rtImguiRPI.renderPass = imguiRenderPass;
        rtImguiRPI.framebuffer = swapChainFramebuffers[imgIdx];
        rtImguiRPI.renderArea.extent = windowExtent;
        rtImguiRPI.clearValueCount = 0;
        vkCmdBeginRenderPass(commandBuffer, &rtImguiRPI, VK_SUBPASS_CONTENTS_INLINE);
        renderInGameUI(commandBuffer, isPlaying); // in-game UI overlay (play mode only)
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
        vkCmdEndRenderPass(commandBuffer);
    }

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &imageAvailableSemaphore;
    si.pWaitDstStageMask = waitStages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &commandBuffer;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &renderFinishedSemaphore;
    vkQueueSubmit(graphicsQueue, 1, &si, inFlightFence);

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderFinishedSemaphore;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapChain;
    pi.pImageIndices = &imgIdx;

    result = vkQueuePresentKHR(graphicsQueue, &pi);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    } else if (result != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare la prezentarea imaginii!");
    }
}


void Renderer::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory, uint32_t mipLevels) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateImage(device, &imageInfo, nullptr, &image);

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, image, &memReqs);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);
    vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory);
    vkBindImageMemory(device, image, imageMemory, 0);
}

void Renderer::generateMipmaps(VkImage image, int32_t texWidth, int32_t texHeight, uint32_t mipLevels) {
    VkCommandBuffer cb = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipW = texWidth, mipH = texHeight;
    for (uint32_t i = 1; i < mipLevels; i++) {
        // Transition level i-1: TRANSFER_DST → TRANSFER_SRC
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipW, mipH, 1};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipW > 1 ? mipW / 2 : 1, mipH > 1 ? mipH / 2 : 1, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        vkCmdBlitImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit, VK_FILTER_LINEAR);

        // Transition level i-1: TRANSFER_SRC → SHADER_READ_ONLY
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        if (mipW > 1) mipW /= 2;
        if (mipH > 1) mipH /= 2;
    }
    // Transition last mip level: TRANSFER_DST → SHADER_READ_ONLY
    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(cb);
}

VkCommandBuffer Renderer::beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

void Renderer::endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void Renderer::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    } else {
        throw std::invalid_argument("Tranzitie de layout nesuportata!");
    }

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(commandBuffer);
}

void Renderer::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(commandBuffer);
}

glm::vec3 Renderer::getRayFromMouse(double mouseX, double mouseY) {
    float x = (2.0f * (float)mouseX) / (float)windowExtent.width - 1.0f;
    float y = 1.0f - (2.0f * (float)mouseY) / (float)windowExtent.height;
    glm::vec4 ray_clip = glm::vec4(x, y, -1.0f, 1.0f);

    glm::mat4 proj = camera->getProjMatrix(windowExtent.width / (float)windowExtent.height);

    glm::vec4 ray_eye = glm::inverse(proj) * ray_clip;
    ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);

    glm::mat4 view = camera->getViewMatrix();
    glm::vec3 ray_wor = glm::vec3(glm::inverse(view) * ray_eye);

    return glm::normalize(ray_wor);
}

bool Renderer::raySphereIntersect(glm::vec3 rayOrigin, glm::vec3 rayDir, glm::vec3 sphereCenter, float sphereRadius, float& hitDistance) {
    glm::vec3 oc = rayOrigin - sphereCenter;
    float b = glm::dot(oc, rayDir);
    float c = glm::dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - c;

    if (discriminant > 0) {
        hitDistance = -b - sqrt(discriminant);
        return hitDistance > 0;
    }
    return false;
}

// --- Particle System ---

void Renderer::destroyEmitterState(EmitterState& es) {
    if (es.buffer != VK_NULL_HANDLE) {
        vkUnmapMemory(device, es.memory);
        vkDestroyBuffer(device, es.buffer, nullptr);
        vkFreeMemory(device, es.memory, nullptr);
        es.buffer = VK_NULL_HANDLE;
        es.memory = VK_NULL_HANDLE;
        es.mapped = nullptr;
    }
}

void Renderer::tickParticles(float dt) {
    // Camera basis for billboard expansion
    glm::vec3 camRight = glm::normalize(glm::cross(camera->front, camera->up));
    glm::vec3 camUp    = camera->up;

    // Collect active emitter entity IDs this frame
    std::set<uint32_t> activeIds;

    for (int ei = 0; ei < (int)scene->entities.size(); ei++) {
        auto& entity = scene->entities[ei];
        if (!entity.hasEmitter) continue;
        activeIds.insert(entity.id);

        EmitterState& es = particleEmitters[entity.id];

        // (Re)allocate buffer if capacity changed or first use
        int needed = std::max(1, entity.emitterMaxParticles);
        if (es.capacity != needed) {
            destroyEmitterState(es);
            es.particles.assign(needed, ParticleCPU{});
            es.capacity = needed;
            VkDeviceSize bufSize = (VkDeviceSize)(needed * 6 * sizeof(ParticleVertex));
            createBuffer(bufSize,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                es.buffer, es.memory);
            vkMapMemory(device, es.memory, 0, bufSize, 0, &es.mapped);
            es.emitAccum = 0.0f;
        }

        // Integrate alive particles
        for (auto& p : es.particles) {
            if (!p.alive) continue;
            p.position += p.velocity * dt;
            p.age      += dt;
            if (p.age >= p.maxAge) p.alive = false;
        }

        // Spawn new particles
        es.emitAccum += entity.emitterRate * dt;
        int toSpawn = (int)es.emitAccum;
        es.emitAccum -= (float)toSpawn;

        glm::vec3 emitterPos = glm::vec3(getWorldTransform(*scene, ei)[3]);

        for (int s = 0; s < toSpawn; s++) {
            // Find a dead slot
            for (auto& p : es.particles) {
                if (p.alive) continue;
                // Random direction within spread cone
                float r1 = (float)rand() / RAND_MAX;
                float r2 = (float)rand() / RAND_MAX;
                float phi   = r1 * 2.0f * 3.14159265f;
                float theta = r2 * entity.emitterSpread;
                glm::vec3 dir = glm::vec3(
                    glm::sin(theta) * glm::cos(phi),
                    glm::cos(theta),
                    glm::sin(theta) * glm::sin(phi));
                p.position = emitterPos;
                p.velocity = dir * entity.emitterSpeed;
                p.age      = 0.0f;
                p.maxAge   = entity.emitterLifetime;
                p.alive    = true;
                break;
            }
        }

        // Rebuild GPU vertex buffer from alive particles
        auto* verts = static_cast<ParticleVertex*>(es.mapped);
        int alive = 0;
        for (const auto& p : es.particles) {
            if (!p.alive || alive >= entity.emitterMaxParticles) continue;
            float t = glm::clamp(p.age / p.maxAge, 0.0f, 1.0f);
            float size = glm::mix(entity.emitterStartSize, entity.emitterEndSize, t);
            glm::vec4 col = glm::mix(entity.emitterStartColor, entity.emitterEndColor, t);

            glm::vec3 r = camRight * size * 0.5f;
            glm::vec3 u = camUp    * size * 0.5f;
            glm::vec3 BL = p.position - r - u;
            glm::vec3 BR = p.position + r - u;
            glm::vec3 TL = p.position - r + u;
            glm::vec3 TR = p.position + r + u;

            ParticleVertex* q = verts + alive * 6;
            q[0] = {BL, {0,0}, col}; q[1] = {BR, {1,0}, col}; q[2] = {TL, {0,1}, col};
            q[3] = {BR, {1,0}, col}; q[4] = {TR, {1,1}, col}; q[5] = {TL, {0,1}, col};
            alive++;
        }
        es.aliveCount = alive;
    }

    // Destroy emitter states for deleted entities
    for (auto it = particleEmitters.begin(); it != particleEmitters.end(); ) {
        if (!activeIds.count(it->first)) {
            destroyEmitterState(it->second);
            it = particleEmitters.erase(it);
        } else {
            ++it;
        }
    }
}

void Renderer::renderParticles(VkCommandBuffer cb) {
    bool anyAlive = false;
    for (auto& [id, es] : particleEmitters)
        if (es.aliveCount > 0) { anyAlive = true; break; }
    if (!anyAlive) return;

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, particlePipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, particlePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

    for (auto& [id, es] : particleEmitters) {
        if (es.aliveCount == 0 || es.buffer == VK_NULL_HANDLE) continue;
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &es.buffer, &offset);
        vkCmdDraw(cb, (uint32_t)(es.aliveCount * 6), 1, 0, 0);
    }
}

// --- Skeletal Animation helpers ---

static glm::vec3 sampleVec3(const std::vector<TranslationKey>& keys, float t) {
    if (keys.empty()) return glm::vec3(0.0f);
    if (keys.size() == 1 || t <= keys.front().time) return keys.front().value;
    if (t >= keys.back().time) return keys.back().value;
    for (size_t i = 0; i + 1 < keys.size(); i++) {
        if (t >= keys[i].time && t < keys[i+1].time) {
            float a = (t - keys[i].time) / (keys[i+1].time - keys[i].time);
            return glm::mix(keys[i].value, keys[i+1].value, a);
        }
    }
    return keys.back().value;
}

static glm::quat sampleQuat(const std::vector<RotationKey>& keys, float t) {
    if (keys.empty()) return glm::quat(1,0,0,0);
    if (keys.size() == 1 || t <= keys.front().time) return keys.front().value;
    if (t >= keys.back().time) return keys.back().value;
    for (size_t i = 0; i + 1 < keys.size(); i++) {
        if (t >= keys[i].time && t < keys[i+1].time) {
            float a = (t - keys[i].time) / (keys[i+1].time - keys[i].time);
            return glm::slerp(keys[i].value, keys[i+1].value, a);
        }
    }
    return keys.back().value;
}

static glm::vec3 sampleScale(const std::vector<ScaleKey>& keys, float t) {
    if (keys.empty()) return glm::vec3(1.0f);
    if (keys.size() == 1 || t <= keys.front().time) return keys.front().value;
    if (t >= keys.back().time) return keys.back().value;
    for (size_t i = 0; i + 1 < keys.size(); i++) {
        if (t >= keys[i].time && t < keys[i+1].time) {
            float a = (t - keys[i].time) / (keys[i+1].time - keys[i].time);
            return glm::mix(keys[i].value, keys[i+1].value, a);
        }
    }
    return keys.back().value;
}

// --- Skeletal Animation ---

void Renderer::destroySkinInstance(uint32_t entityId) {
    auto it = skinInstances.find(entityId);
    if (it == skinInstances.end()) return;
    SkinInstance& si = it->second;
    if (si.descSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device, skinDescPool, 1, &si.descSet);
        si.descSet = VK_NULL_HANDLE;
    }
    if (si.ssbo != VK_NULL_HANDLE) {
        vkUnmapMemory(device, si.mem);
        vkDestroyBuffer(device, si.ssbo, nullptr);
        vkFreeMemory(device, si.mem, nullptr);
        si.ssbo   = VK_NULL_HANDLE;
        si.mem    = VK_NULL_HANDLE;
        si.mapped = nullptr;
    }
    skinInstances.erase(it);
}

void Renderer::tickSkinning(float dt) {
    for (auto& entity : scene->entities) {
        if (!entity.hasSkin || entity.skinModelName.empty()) continue;

        // Load skinned mesh + skin data on first encounter
        if (!loadedSkinnedMeshes.count(entity.skinModelName))
            loadSkinnedMesh(entity.skinModelName);
        if (!loadedSkins.count(entity.skinModelName)) continue;

        const SkinData& skin = loadedSkins[entity.skinModelName];
        if (skin.numJoints == 0) continue;

        // Sync SkinInstance properties from entity
        SkinInstance& si = skinInstances[entity.id];
        si.playing          = entity.animationPlaying;
        si.loop             = entity.animationLoop;
        si.speed            = entity.animationSpeed;
        if (si.currentAnimation != entity.currentAnimation) {
            si.currentAnimation = entity.currentAnimation;
            si.playbackTime     = 0.0f;
        }

        // Allocate or reallocate SSBO if joint count changed
        if (si.ssboJointCount != skin.numJoints) {
            if (si.ssbo != VK_NULL_HANDLE) {
                vkUnmapMemory(device, si.mem);
                vkDestroyBuffer(device, si.ssbo, nullptr);
                vkFreeMemory(device, si.mem, nullptr);
            }
            VkDeviceSize bufSize = sizeof(glm::mat4) * skin.numJoints;
            createBuffer(bufSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                si.ssbo, si.mem);
            vkMapMemory(device, si.mem, 0, bufSize, 0, &si.mapped);
            si.ssboJointCount = skin.numJoints;

            // Allocate descriptor set from skinDescPool
            VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            allocInfo.descriptorPool     = skinDescPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts        = &skinDescSetLayout;
            vkAllocateDescriptorSets(device, &allocInfo, &si.descSet);

            VkDescriptorBufferInfo bufInfo{si.ssbo, 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet         = si.descSet;
            write.dstBinding     = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.descriptorCount = 1;
            write.pBufferInfo    = &bufInfo;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        // Advance playback time
        if (si.playing) {
            const int animIdx = std::clamp(si.currentAnimation, 0, (int)skin.animations.size() - 1);
            const AnimationClip& clip = skin.animations[animIdx];
            si.playbackTime += dt * si.speed;
            if (clip.duration > 0.0f) {
                if (si.playbackTime > clip.duration) {
                    if (si.loop) si.playbackTime = fmodf(si.playbackTime, clip.duration);
                    else         si.playbackTime = clip.duration;
                }
            }
        }

        // Compute joint matrices
        const int animIdx = std::clamp(si.currentAnimation, 0, (int)skin.animations.size() - 1);
        const AnimationClip& clip = skin.animations[animIdx];
        float t = si.playbackTime;

        // Build per-joint local TRS matrices from animation channels (or rest pose)
        std::vector<glm::mat4> localMats(skin.numJoints);
        // Start from rest poses
        for (int j = 0; j < skin.numJoints; j++) localMats[j] = skin.localBindPoses[j];

        for (const auto& track : clip.tracks) {
            int ji = track.jointIndex;
            if (ji < 0 || ji >= skin.numJoints) continue;
            glm::vec3 tr  = track.translations.empty() ? glm::vec3(skin.localBindPoses[ji][3])
                                                        : sampleVec3(track.translations, t);
            glm::quat rot = track.rotations.empty()    ? glm::quat_cast(skin.localBindPoses[ji])
                                                        : sampleQuat(track.rotations, t);
            glm::vec3 sc  = track.scales.empty()       ? glm::vec3(1.0f)
                                                        : sampleScale(track.scales, t);
            localMats[ji] = glm::translate(glm::mat4(1.0f), tr)
                          * glm::mat4_cast(rot)
                          * glm::scale(glm::mat4(1.0f), sc);
        }

        // Compute global joint matrices (parent-first order — GLTF guarantees parents before children)
        std::vector<glm::mat4> globalMats(skin.numJoints);
        for (int j = 0; j < skin.numJoints; j++) {
            if (skin.parentJoint[j] < 0)
                globalMats[j] = localMats[j];
            else
                globalMats[j] = globalMats[skin.parentJoint[j]] * localMats[j];
        }

        // Final joint matrix = globalJointMat * inverseBindMatrix
        glm::mat4* dst = static_cast<glm::mat4*>(si.mapped);
        for (int j = 0; j < skin.numJoints; j++)
            dst[j] = globalMats[j] * skin.inverseBindMatrices[j];
    }

    // Clean up instances for entities that no longer have hasSkin
    std::set<uint32_t> activeIds;
    for (const auto& e : scene->entities)
        if (e.hasSkin) activeIds.insert(e.id);
    for (auto it = skinInstances.begin(); it != skinInstances.end(); ) {
        if (!activeIds.count(it->first)) {
            SkinInstance& si = it->second;
            if (si.descSet != VK_NULL_HANDLE)
                vkFreeDescriptorSets(device, skinDescPool, 1, &si.descSet);
            if (si.ssbo != VK_NULL_HANDLE) {
                vkUnmapMemory(device, si.mem);
                vkDestroyBuffer(device, si.ssbo, nullptr);
                vkFreeMemory(device, si.mem, nullptr);
            }
            it = skinInstances.erase(it);
        } else { ++it; }
    }
}

void Renderer::renderSkinned(VkCommandBuffer cb) {
    bool any = false;
    for (const auto& e : scene->entities)
        if (e.hasSkin && !e.skinModelName.empty() && loadedSkinnedMeshes.count(e.skinModelName)) { any = true; break; }
    if (!any) return;

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, skinnedPipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, skinnedPipelineLayout,
        0, 1, &descriptorSet, 0, nullptr);

    for (int i = 0; i < (int)scene->entities.size(); i++) {
        const auto& entity = scene->entities[i];
        if (!entity.hasSkin || entity.skinModelName.empty()) continue;

        auto meshIt = loadedSkinnedMeshes.find(entity.skinModelName);
        if (meshIt == loadedSkinnedMeshes.end()) continue;
        auto siIt = skinInstances.find(entity.id);
        if (siIt == skinInstances.end() || siIt->second.ssbo == VK_NULL_HANDLE) continue;

        // Bind per-entity SSBO at set=1
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, skinnedPipelineLayout,
            1, 1, &siIt->second.descSet, 0, nullptr);

        SimplePushConstantData push{};
        push.modelMatrix  = getWorldTransform(*scene, i);
        push.textureIndex = std::clamp(entity.textureIndex, 0,
                               textureImages.empty() ? 0 : (int)textureImages.size() - 1);
        push.metallic     = entity.metallic;
        push.roughness    = entity.roughness;
        vkCmdPushConstants(cb, skinnedPipelineLayout,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0, sizeof(SimplePushConstantData), &push);

        const SkinnedMeshResource& mesh = meshIt->second;
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &mesh.vertexBuffer, &offset);
        vkCmdBindIndexBuffer(cb, mesh.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, mesh.indexCount, 1, 0, 0, 0);
    }
}

// --- Play Mode ---


Navmesh Renderer::buildNavmesh() {
    static constexpr float kCellSize    = 0.5f;
    static constexpr float kPadding     = 5.0f;
    static constexpr float kAgentHeight = 1.8f;

    float minX = -20.0f, maxX = 20.0f;
    float minY = -20.0f, maxY = 20.0f;
    for (const auto& e : scene->entities) {
        glm::vec3 p = e.transform.translation;
        minX = std::min(minX, p.x - e.transform.scale.x);
        maxX = std::max(maxX, p.x + e.transform.scale.x);
        minY = std::min(minY, p.y - e.transform.scale.y);
        maxY = std::max(maxY, p.y + e.transform.scale.y);
    }
    minX -= kPadding; minY -= kPadding;
    maxX += kPadding; maxY += kPadding;

    Navmesh navmesh;
    navmesh.cellSize = kCellSize;
    navmesh.originX  = minX;
    navmesh.originY  = minY;
    navmesh.width    = std::max(1, (int)((maxX - minX) / kCellSize));
    navmesh.depth    = std::max(1, (int)((maxY - minY) / kCellSize));
    navmesh.cells.assign(navmesh.width * navmesh.depth, NavCell{true});

    // 2D point-in-triangle test
    auto triSign = [](glm::vec2 p, glm::vec2 a, glm::vec2 b) {
        return (p.x - b.x) * (a.y - b.y) - (a.x - b.x) * (p.y - b.y);
    };
    auto pointInTri2D = [&](glm::vec2 pt, glm::vec2 a, glm::vec2 b, glm::vec2 c) {
        float d1 = triSign(pt, a, b);
        float d2 = triSign(pt, b, c);
        float d3 = triSign(pt, c, a);
        bool hasNeg = (d1 < 0) || (d2 < 0) || (d3 < 0);
        bool hasPos = (d1 > 0) || (d2 > 0) || (d3 > 0);
        return !(hasNeg && hasPos);
    };

    for (int ei = 0; ei < (int)scene->entities.size(); ei++) {
        const auto& e = scene->entities[ei];
        if (!e.isStatic || !e.hasCollision || e.isTerrain || e.isLightSource) continue;

        // Resolve mesh key
        std::string meshKey;
        if (!availableModels.empty() && e.modelIndex >= 0 &&
            (size_t)e.modelIndex < availableModels.size())
            meshKey = availableModels[e.modelIndex];

        auto meshIt = meshKey.empty() ? loadedMeshes.end() : loadedMeshes.find(meshKey);
        bool hasMesh = (meshIt != loadedMeshes.end()) && !meshIt->second.cpuVerts.empty();

        if (hasMesh) {
            const MeshResource& mesh = meshIt->second;
            glm::mat4 worldMat  = getWorldTransform(*scene, ei);
            float groundZ = e.transform.translation.z - e.transform.scale.z * 0.5f;
            float ceilZ   = groundZ + kAgentHeight;

            for (size_t ti = 0; ti + 2 < mesh.cpuIdx.size(); ti += 3) {
                glm::vec3 w0 = glm::vec3(worldMat * glm::vec4(mesh.cpuVerts[mesh.cpuIdx[ti  ]].pos, 1.f));
                glm::vec3 w1 = glm::vec3(worldMat * glm::vec4(mesh.cpuVerts[mesh.cpuIdx[ti+1]].pos, 1.f));
                glm::vec3 w2 = glm::vec3(worldMat * glm::vec4(mesh.cpuVerts[mesh.cpuIdx[ti+2]].pos, 1.f));

                // Skip triangles fully outside the agent-height column
                float zMin = std::min({w0.z, w1.z, w2.z});
                float zMax = std::max({w0.z, w1.z, w2.z});
                if (zMax < groundZ || zMin > ceilZ) continue;

                glm::vec2 a = {w0.x, w0.y};
                glm::vec2 b = {w1.x, w1.y};
                glm::vec2 c = {w2.x, w2.y};

                float txMin = std::min({a.x, b.x, c.x});
                float txMax = std::max({a.x, b.x, c.x});
                float tyMin = std::min({a.y, b.y, c.y});
                float tyMax = std::max({a.y, b.y, c.y});

                int cxMin = std::max(0, (int)((txMin - navmesh.originX) / kCellSize));
                int cxMax = std::min(navmesh.width-1, (int)((txMax - navmesh.originX) / kCellSize));
                int cyMin = std::max(0, (int)((tyMin - navmesh.originY) / kCellSize));
                int cyMax = std::min(navmesh.depth-1, (int)((tyMax - navmesh.originY) / kCellSize));

                for (int cy = cyMin; cy <= cyMax; cy++)
                    for (int cx = cxMin; cx <= cxMax; cx++)
                        if (pointInTri2D(navmesh.worldPos(cx, cy), a, b, c))
                            navmesh.cells[navmesh.idx(cx, cy)].walkable = false;
            }
        } else {
            // No CPU mesh data — fall back to transform-scale AABB
            float ex = e.transform.translation.x;
            float ey = e.transform.translation.y;
            float hw = e.transform.scale.x * 0.5f;
            float hd = e.transform.scale.y * 0.5f;

            int cxMin = std::max(0, (int)((ex - hw - navmesh.originX) / kCellSize));
            int cxMax = std::min(navmesh.width-1, (int)((ex + hw - navmesh.originX) / kCellSize));
            int cyMin = std::max(0, (int)((ey - hd - navmesh.originY) / kCellSize));
            int cyMax = std::min(navmesh.depth-1, (int)((ey + hd - navmesh.originY) / kCellSize));

            for (int cy = cyMin; cy <= cyMax; cy++)
                for (int cx = cxMin; cx <= cxMax; cx++)
                    navmesh.cells[navmesh.idx(cx, cy)].walkable = false;
        }
    }

    int blocked = 0;
    for (const auto& c : navmesh.cells) if (!c.walkable) blocked++;
    LOG_INFO("Navmesh built: %dx%d cells, %d blocked (mesh-accurate)", navmesh.width, navmesh.depth, blocked);
    return navmesh;
}

// ---------------------------------------------------------------------------
// In-game UI rendering helpers
// ---------------------------------------------------------------------------

// Convert pixel position to Vulkan NDC [-1,1] with Y down matching screen space
static glm::vec2 pixelToNDC(float px, float py, float sw, float sh) {
    return { (px / sw) * 2.0f - 1.0f, (py / sh) * 2.0f - 1.0f };
}

// Anchor → pixel origin (top-left corner is 0,0)
static glm::vec2 anchorOrigin(UIAnchor anchor, float sw, float sh) {
    float ax = 0.0f, ay = 0.0f;
    switch (anchor) {
    case UIAnchor::TopLeft:      ax = 0;    ay = 0;    break;
    case UIAnchor::TopCenter:    ax = sw/2; ay = 0;    break;
    case UIAnchor::TopRight:     ax = sw;   ay = 0;    break;
    case UIAnchor::MiddleLeft:   ax = 0;    ay = sh/2; break;
    case UIAnchor::Center:       ax = sw/2; ay = sh/2; break;
    case UIAnchor::MiddleRight:  ax = sw;   ay = sh/2; break;
    case UIAnchor::BottomLeft:   ax = 0;    ay = sh;   break;
    case UIAnchor::BottomCenter: ax = sw/2; ay = sh;   break;
    case UIAnchor::BottomRight:  ax = sw;   ay = sh;   break;
    }
    return {ax, ay};
}

// Emit a solid-color quad (2 triangles, 6 vertices) — mode 0
static void emitSolidQuad(std::vector<UIVertex>& v,
                          glm::vec2 p0, glm::vec2 p1, glm::vec4 color) {
    v.push_back({ p0,              {0,0}, color });
    v.push_back({ {p1.x,p0.y},    {0,0}, color });
    v.push_back({ p1,              {0,0}, color });
    v.push_back({ p0,              {0,0}, color });
    v.push_back({ p1,              {0,0}, color });
    v.push_back({ {p0.x,p1.y},    {0,0}, color });
}

// UV for character c in the 128×64 font atlas (16 chars/row, 8 rows, each 8×8)
static glm::vec4 charUV(int c) {
    if (c < 32 || c > 127) c = 32; // fallback to space
    int col = c % 16, row = c / 16;
    float u0 = col / 16.0f, v0 = row / 8.0f;
    return { u0, v0, u0 + 1/16.0f, v0 + 1/8.0f };
}

// Emit a single glyph quad — mode 1
static void emitGlyphQuad(std::vector<UIVertex>& v,
                          glm::vec2 p0, glm::vec2 p1, int c, glm::vec4 color) {
    glm::vec4 uv = charUV(c);
    glm::vec2 uv0{uv.x, uv.y}, uv1{uv.z, uv.w};
    v.push_back({ p0,            uv0,           color });
    v.push_back({ {p1.x,p0.y},  {uv1.x,uv0.y}, color });
    v.push_back({ p1,            uv1,           color });
    v.push_back({ p0,            uv0,           color });
    v.push_back({ p1,            uv1,           color });
    v.push_back({ {p0.x,p1.y},  {uv0.x,uv1.y}, color });
}

// Build text quads into the font vertex list
static void emitText(std::vector<UIVertex>& fontVerts,
                     const std::string& text,
                     glm::vec2 pixOrig, float charW, float charH,
                     float sw, float sh, glm::vec4 color) {
    float cx = pixOrig.x;
    for (char ch : text) {
        glm::vec2 p0 = pixelToNDC(cx,       pixOrig.y,        sw, sh);
        glm::vec2 p1 = pixelToNDC(cx+charW, pixOrig.y+charH,  sw, sh);
        emitGlyphQuad(fontVerts, p0, p1, (int)(unsigned char)ch, color);
        cx += charW;
    }
}

void Renderer::renderInGameUI(VkCommandBuffer cb, bool isPlaying) {
    if (!isPlaying || uiPipeline == VK_NULL_HANDLE || uiVertexMapped == nullptr) return;

    float sw = (float)windowExtent.width;
    float sh = (float)windowExtent.height;
    const float kCharW = 8.0f, kCharH = 8.0f, kScale = 2.0f;
    const float cw = kCharW * kScale, ch = kCharH * kScale;

    // Separate solid and font vertex batches
    std::vector<UIVertex> solidVerts, fontVerts;
    solidVerts.reserve(256);
    fontVerts.reserve(1024);

    for (const auto& entity : scene->entities) {
        if (!entity.hasUICanvas || !entity.uiCanvas.visible) continue;

        for (const auto& el : entity.uiCanvas.elements) {
            glm::vec2 orig = anchorOrigin(el.anchor, sw, sh) + el.offset;
            glm::vec2 p0   = pixelToNDC(orig.x,           orig.y,           sw, sh);
            glm::vec2 p1   = pixelToNDC(orig.x + el.size.x, orig.y + el.size.y, sw, sh);

            switch (el.type) {
            case UIElementType::Label:
                // Semi-transparent background + text
                if (el.bgColor.a > 0.0f)
                    emitSolidQuad(solidVerts, p0, p1, el.bgColor);
                emitText(fontVerts, el.text,
                         {orig.x + 4.0f, orig.y + (el.size.y - ch) * 0.5f},
                         cw, ch, sw, sh, el.color);
                break;

            case UIElementType::Button:
                emitSolidQuad(solidVerts, p0, p1, el.bgColor);
                // 1px darker border approximated by slightly smaller filled rect
                {
                    glm::vec2 bp0 = pixelToNDC(orig.x+1, orig.y+1, sw, sh);
                    glm::vec2 bp1 = pixelToNDC(orig.x+el.size.x-1, orig.y+el.size.y-1, sw, sh);
                    glm::vec4 fillColor = el.bgColor + glm::vec4(0.15f,0.15f,0.15f,0.0f);
                    emitSolidQuad(solidVerts, bp0, bp1, fillColor);
                }
                emitText(fontVerts, el.text,
                         {orig.x + (el.size.x - el.text.size() * cw) * 0.5f,
                          orig.y + (el.size.y - ch) * 0.5f},
                         cw, ch, sw, sh, el.color);
                break;

            case UIElementType::Healthbar:
                // Background
                emitSolidQuad(solidVerts, p0, p1, el.bgColor);
                // Fill (clamped to [0,1])
                {
                    float fill = std::max(0.0f, std::min(1.0f, el.value));
                    glm::vec2 fp1 = pixelToNDC(orig.x + el.size.x * fill, orig.y + el.size.y, sw, sh);
                    emitSolidQuad(solidVerts, p0, fp1, el.valueColor);
                }
                // Optional label
                if (!el.text.empty())
                    emitText(fontVerts, el.text,
                             {orig.x + (el.size.x - el.text.size() * cw) * 0.5f,
                              orig.y + (el.size.y - ch) * 0.5f},
                             cw, ch, sw, sh, el.color);
                break;

            case UIElementType::Image:
                // Just show a placeholder colored rect for now
                emitSolidQuad(solidVerts, p0, p1, el.color);
                break;
            }
        }
    }

    int totalVerts = (int)(solidVerts.size() + fontVerts.size());
    if (totalVerts == 0 || totalVerts > kUIMaxVertices) return;

    // Upload: solid batch first, then font batch
    UIVertex* mapped = reinterpret_cast<UIVertex*>(uiVertexMapped);
    int solidCount = (int)solidVerts.size();
    int fontCount  = (int)fontVerts.size();
    if (solidCount > 0) memcpy(mapped,              solidVerts.data(), solidCount * sizeof(UIVertex));
    if (fontCount  > 0) memcpy(mapped + solidCount, fontVerts.data(),  fontCount  * sizeof(UIVertex));

    // Set viewport + scissor
    VkViewport vp{0.0f, 0.0f, sw, sh, 0.0f, 1.0f};
    VkRect2D   sc{{0,0}, windowExtent};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, uiPipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, uiPipelineLayout, 0, 1, &uiDescSet, 0, nullptr);
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &uiVertexBuffer, &offset);

    // Solid batch (mode=0)
    if (solidCount > 0) {
        int mode = 0;
        vkCmdPushConstants(cb, uiPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &mode);
        vkCmdDraw(cb, solidCount, 1, 0, 0);
    }
    // Font batch (mode=1)
    if (fontCount > 0) {
        int mode = 1;
        vkCmdPushConstants(cb, uiPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(int), &mode);
        vkCmdDraw(cb, fontCount, 1, solidCount, 0);
    }
}

void Renderer::tickInGameUI(bool isPlaying) {
    if (!isPlaying) return;

    // Detect left-mouse-button press (rising edge only — uses member so it resets on play/stop)
    bool curMouseDown = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool justClicked  = curMouseDown && !uiPrevMouseDown;
    uiPrevMouseDown = curMouseDown;
    if (!justClicked) return;

    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    float sw = (float)windowExtent.width;
    float sh = (float)windowExtent.height;

    for (auto& entity : scene->entities) {
        if (!entity.hasUICanvas || !entity.uiCanvas.visible) continue;

        for (auto& el : entity.uiCanvas.elements) {
            if (el.type != UIElementType::Button) continue;
            if (!el.onClick) continue;

            glm::vec2 orig = anchorOrigin(el.anchor, sw, sh) + el.offset;
            float x0 = orig.x, y0 = orig.y;
            float x1 = x0 + el.size.x, y1 = y0 + el.size.y;

            if (mx >= x0 && mx <= x1 && my >= y0 && my <= y1)
                el.onClick();
        }
    }
}

// --- Dialogue Overlay ---

