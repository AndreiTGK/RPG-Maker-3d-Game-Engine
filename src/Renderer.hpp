#pragma once
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include "Camera.hpp"
#include "Scene.hpp"
#include "Project.hpp"
#include "NavMesh.hpp"
#include "RayTracer.hpp"
#include <vector>
#include <string>
#include <array>
#include <map>
#include <tuple>
#include <functional>

// ---------------------------------------------------------------------------
// GPU vertex / data structs
// ---------------------------------------------------------------------------

struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bd{};
        bd.binding = 0;
        bd.stride  = sizeof(Vertex);
        bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bd;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 4> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)};
        a[2] = {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)};
        a[3] = {3, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, texCoord)};
        return a;
    }
};

struct GpuPointLight {
    alignas(16) glm::vec3 position;
    float intensity;
    alignas(16) glm::vec3 color;
    float radius;
};
static_assert(sizeof(GpuPointLight) == 32, "GpuPointLight size mismatch");

struct UniformBufferObject {
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 lightSpaceMatrix[4];
    alignas(16) glm::vec3 ambientLight;
    float _pad0 = 0.0f;
    alignas(16) glm::vec3 sunDirection;
    float shadowsEnabled = 1.0f;
    alignas(16) glm::vec3 skyColor;
    int numActiveLights = 0;
    alignas(16) GpuPointLight lights[4];
};
static_assert(offsetof(UniformBufferObject, lights) == 432, "UBO lights offset mismatch");
static_assert(sizeof(UniformBufferObject) == 560, "UBO size mismatch");

struct ShadowPushConstant {
    glm::mat4 modelMatrix;
    glm::mat4 lightSpaceMatrix;
};

// Omnidirectional point-light shadow: stores lightPos+radius alongside the face matrix.
// 144 bytes — within the 256-byte push constant limit supported by all Vulkan 1.2 desktop GPUs.
struct CubeShadowPushConstant {
    glm::mat4 modelMatrix;    // 64
    glm::mat4 faceViewProj;   // 64
    glm::vec4 lightPosRadius; // 16  (xyz = world pos, w = radius)
};

static constexpr int kUnlitTextureIndex = -1;

struct SimplePushConstantData {
    glm::mat4 modelMatrix;
    int   textureIndex = 0;
    float metallic     = 0.0f;
    float roughness    = 0.5f;
};

struct UIVertex {
    glm::vec2 pos;
    glm::vec2 uv;
    glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bd{};
        bd.binding = 0; bd.stride = sizeof(UIVertex); bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bd;
    }
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(UIVertex, uv)};
        a[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(UIVertex, color)};
        return a;
    }
};

struct ParticleVertex {
    glm::vec3 position;
    glm::vec2 uv;
    glm::vec4 color;
};

struct ParticleCPU {
    glm::vec3 position;
    glm::vec3 velocity;
    float     age    = 0.0f;
    float     maxAge = 1.0f;
    bool      alive  = false;
};

struct EmitterState {
    std::vector<ParticleCPU> particles;
    VkBuffer       buffer    = VK_NULL_HANDLE;
    VkDeviceMemory memory    = VK_NULL_HANDLE;
    void*          mapped    = nullptr;
    float          emitAccum = 0.0f;
    int            aliveCount = 0;
    int            capacity   = 0;
};

struct SkinnedVertex {
    glm::vec3  pos;
    glm::vec3  normal;
    glm::vec2  texCoord;
    glm::uvec4 joints;
    glm::vec4  weights;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bd{};
        bd.binding = 0; bd.stride = sizeof(SkinnedVertex); bd.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bd;
    }
    static std::array<VkVertexInputAttributeDescription, 5> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 5> a{};
        a[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(SkinnedVertex, pos)};
        a[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT,    offsetof(SkinnedVertex, normal)};
        a[2] = {2, 0, VK_FORMAT_R32G32_SFLOAT,       offsetof(SkinnedVertex, texCoord)};
        a[3] = {3, 0, VK_FORMAT_R32G32B32A32_UINT,   offsetof(SkinnedVertex, joints)};
        a[4] = {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(SkinnedVertex, weights)};
        return a;
    }
};

struct TranslationKey { float time; glm::vec3 value; };
struct RotationKey    { float time; glm::quat value; };
struct ScaleKey       { float time; glm::vec3 value; };

struct JointTrack {
    int jointIndex = 0;
    std::vector<TranslationKey> translations;
    std::vector<RotationKey>    rotations;
    std::vector<ScaleKey>       scales;
};

struct AnimationClip {
    std::string name;
    float duration = 0.0f;
    std::vector<JointTrack> tracks;
};

struct SkinData {
    int numJoints = 0;
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<int>       parentJoint;
    std::vector<glm::mat4> localBindPoses;
    std::vector<AnimationClip> animations;
};

struct SkinnedMeshResource {
    VkBuffer       vertexBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer       indexBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory  = VK_NULL_HANDLE;
    uint32_t       indexCount         = 0;
    uint32_t       vertexCount        = 0;
};

struct SkinInstance {
    int   currentAnimation = 0;
    float playbackTime     = 0.0f;
    bool  loop             = true;
    bool  playing          = false;
    float speed            = 1.0f;
    VkBuffer        ssbo    = VK_NULL_HANDLE;
    VkDeviceMemory  mem     = VK_NULL_HANDLE;
    void*           mapped  = nullptr;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    int             ssboJointCount = 0;
};

// ---------------------------------------------------------------------------
// Renderer — owns all GPU state: Vulkan context, pipelines, resource maps,
// and the full render loop. VulkanEngine holds one Renderer and delegates
// all GPU work through it.
// ---------------------------------------------------------------------------

class Renderer {
public:
    // --- Public data (accessed by VulkanEngine for project/editor integration) ---

    // Asset lists populated by scan methods; Editor holds pointers to these.
    std::vector<std::string> availableModels;
    std::vector<std::string> availableTextures;
    std::vector<std::string> availableAudio;
    int selectedModelIndex  = 0;
    int selectedTextureIndex = 0;

    bool useRayTracing      = false;
    bool framebufferResized = false;
    bool uiPrevMouseDown    = false;  // rising-edge for in-game UI buttons

    VkExtent2D windowExtent{};
    RayTracer  rayTracer;

    struct PostSettings {
        float exposure       = 1.0f;
        float bloomThreshold = 1.0f;
        float bloomStrength  = 0.05f;
    };
    PostSettings postSettings;

    struct MeshResource {
        VkBuffer       vertexBuffer       = VK_NULL_HANDLE;
        VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
        VkBuffer       indexBuffer        = VK_NULL_HANDLE;
        VkDeviceMemory indexBufferMemory  = VK_NULL_HANDLE;
        uint32_t       indexCount         = 0;
        uint32_t       vertexCount        = 0;
        VkBuffer       lodVB[2]           = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkDeviceMemory lodVBMem[2]        = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkBuffer       lodIB[2]           = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkDeviceMemory lodIBMem[2]        = {VK_NULL_HANDLE, VK_NULL_HANDLE};
        uint32_t       lodIndexCount[2]   = {0, 0};
        std::vector<Vertex>   cpuVerts;
        std::vector<uint32_t> cpuIdx;
    };

    std::map<std::string, MeshResource>        loadedMeshes;
    std::map<std::string, SkinData>            loadedSkins;
    std::map<std::string, SkinnedMeshResource> loadedSkinnedMeshes;
    std::map<uint32_t,    SkinInstance>        skinInstances;
    std::map<uint32_t,    EmitterState>        particleEmitters;

    // Exposed so VulkanEngine::loadScene/switchProject can call vkDeviceWaitIdle.
    VkDevice device = VK_NULL_HANDLE;

    // --- Lifecycle ---
    void init(GLFWwindow* w, Scene* s, Camera* c, Workspace* ws);
    void initImGui();
    void drawFrame(float deltaTime, bool isPlaying);
    void cleanup();
    void cleanupSwapChain();
    void recreateSwapChain();

    // --- Asset management ---
    void scanModelsFolder();
    void scanTexturesFolder();
    void scanAudioFolder();
    void loadMesh(const std::string& modelName);
    void loadSkinnedMesh(const std::string& modelName);
    void loadAllProjectMeshes();
    void createFallbackMesh();
    void generateTerrainMesh(const GameObject& entity);
    void destroyMesh(const std::string& key);
    void reloadCurrentModel();
    void reloadCurrentTexture(int selectedEntityIndex);
    void reloadProjectTextures();
    void updateTextureDescriptorSet();
    void packAssets();

    // --- Entity GPU lifecycle ---
    void destroyEmitterState(EmitterState& es);
    void destroySkinInstance(uint32_t entityId);

    // --- Navmesh (needs loadedMeshes CPU data) ---
    Navmesh buildNavmesh();

    // --- Picking ---
    glm::vec3 getRayFromMouse(double mouseX, double mouseY);
    bool raySphereIntersect(glm::vec3 origin, glm::vec3 dir,
                            glm::vec3 center, float radius, float& dist);

    // --- Play-mode UI tick (called before drawFrame each frame) ---
    void tickInGameUI(bool isPlaying);

private:
    GLFWwindow* window    = nullptr;
    Scene*      scene     = nullptr;
    Camera*     camera    = nullptr;
    Workspace*  workspace = nullptr;

    VkInstance       instance       = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkSurfaceKHR     surface        = VK_NULL_HANDLE;
    VkQueue          graphicsQueue;
    uint32_t         graphicsQueueIndex = 0;

    VkSwapchainKHR             swapChain;
    std::vector<VkImage>       swapChainImages;
    std::vector<VkImageView>   swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;

    VkRenderPass renderPass      = VK_NULL_HANDLE;
    VkRenderPass postRenderPass  = VK_NULL_HANDLE;
    VkRenderPass imguiRenderPass = VK_NULL_HANDLE;
    VkRenderPass shadowRenderPass = VK_NULL_HANDLE;

    // 2D shadow atlas — used for the directional sun shadow (tile 0 = 2048×2048 region)
    VkImage        shadowImage       = VK_NULL_HANDLE;
    VkDeviceMemory shadowImageMemory = VK_NULL_HANDLE;
    VkImageView    shadowImageView   = VK_NULL_HANDLE;
    VkSampler      shadowSampler     = VK_NULL_HANDLE;
    VkFramebuffer  shadowFramebuffer = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       shadowPipeline       = VK_NULL_HANDLE;

    // Omnidirectional point-light shadows: R32_SFLOAT cubemap array (4 lights × 6 faces, 512×512 each)
    static constexpr uint32_t kCubeShadowRes = 512;
    static constexpr uint32_t kMaxCubeLights = 4;
    VkImage        shadowCubeImage            = VK_NULL_HANDLE;
    VkDeviceMemory shadowCubeMemory           = VK_NULL_HANDLE;
    VkImageView    shadowCubeArrayView        = VK_NULL_HANDLE;
    std::array<VkImageView,   kMaxCubeLights * 6> shadowCubeFaceViews = {};
    VkImage        shadowCubeDepth            = VK_NULL_HANDLE;
    VkDeviceMemory shadowCubeDepthMem         = VK_NULL_HANDLE;
    VkImageView    shadowCubeDepthView        = VK_NULL_HANDLE;
    std::array<VkFramebuffer, kMaxCubeLights * 6> shadowCubeFBs       = {};
    VkSampler      shadowCubeSampler          = VK_NULL_HANDLE;
    VkRenderPass   shadowCubeRenderPass       = VK_NULL_HANDLE;
    VkPipeline     shadowCubePipeline         = VK_NULL_HANDLE;
    VkPipelineLayout shadowCubePipelineLayout = VK_NULL_HANDLE;

    VkImage        hdrImage       = VK_NULL_HANDLE;
    VkDeviceMemory hdrImageMemory = VK_NULL_HANDLE;
    VkImageView    hdrImageView   = VK_NULL_HANDLE;
    VkSampler      hdrSampler     = VK_NULL_HANDLE;
    VkFramebuffer  hdrFramebuffer = VK_NULL_HANDLE;

    VkDescriptorSetLayout postDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      postDescriptorPool      = VK_NULL_HANDLE;
    VkDescriptorSet       postDescriptorSet       = VK_NULL_HANDLE;
    VkPipelineLayout      postPipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            postPipeline            = VK_NULL_HANDLE;

    VkImage        depthImage       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView    depthImageView   = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            graphicsPipeline    = VK_NULL_HANDLE;
    VkPipeline            linePipeline        = VK_NULL_HANDLE;
    VkDescriptorPool      descriptorPool      = VK_NULL_HANDLE;
    VkDescriptorSet       descriptorSet       = VK_NULL_HANDLE;
    VkDescriptorPool      imguiPool           = VK_NULL_HANDLE;

    VkBuffer       uniformBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory uniformBufferMemory = VK_NULL_HANDLE;
    void*          uniformBufferMapped = nullptr;
    glm::mat4      cachedLightSpaceMatrices[4] = {
        glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f)
    };

    VkCommandPool   commandPool   = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    VkSemaphore imageAvailableSemaphore = VK_NULL_HANDLE;
    VkSemaphore renderFinishedSemaphore = VK_NULL_HANDLE;
    VkFence     inFlightFence           = VK_NULL_HANDLE;

    std::vector<VkImage>        textureImages;
    std::vector<VkDeviceMemory> textureImageMemories;
    std::vector<VkImageView>    textureImageViews;
    std::vector<VkSampler>      textureSamplers;
    std::vector<uint32_t>       textureMipLevels;
    float maxSamplerAnisotropy = 1.0f;

    VkPipelineLayout particlePipelineLayout = VK_NULL_HANDLE;
    VkPipeline       particlePipeline       = VK_NULL_HANDLE;

    VkDescriptorSetLayout skinDescSetLayout    = VK_NULL_HANDLE;
    VkDescriptorPool      skinDescPool         = VK_NULL_HANDLE;
    VkPipelineLayout      skinnedPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            skinnedPipeline       = VK_NULL_HANDLE;

    static constexpr int kUIMaxVertices = 16384;
    VkImage         fontAtlasImage   = VK_NULL_HANDLE;
    VkDeviceMemory  fontAtlasMemory  = VK_NULL_HANDLE;
    VkImageView     fontAtlasView    = VK_NULL_HANDLE;
    VkSampler       fontAtlasSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout uiDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      uiDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet       uiDescSet       = VK_NULL_HANDLE;
    VkPipelineLayout      uiPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            uiPipeline       = VK_NULL_HANDLE;
    VkBuffer       uiVertexBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory uiVertexBufferMemory = VK_NULL_HANDLE;
    void*          uiVertexMapped       = nullptr;

    MeshResource gridMesh;

    std::vector<Vertex>   vertices;
    std::vector<uint32_t> indices;
    VkBuffer       vertexBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer       indexBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory  = VK_NULL_HANDLE;

    float currentScale = 1.0f;

    // --- Setup (VulkanSetup.cpp) ---
    uint32_t      findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    std::vector<char> readFile(const std::string& filename);
    void createInstance();
    void createSurface();
    void setupPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createDepthResources();
    void createRenderPass();
    void createFramebuffers();
    void createHDRResources();
    void createPostRenderPass();
    void createImguiRenderPass();
    void createPostPipeline();
    void createShadowPipeline();
    void createGraphicsPipeline();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createUniformBuffer();
    void createCommandPool();
    void createCommandBuffer();
    void createSyncObjects();
    void createParticlePipeline();
    void createSkinnedPipeline();
    void createShadowResources();
    void createShadowRenderPass();
    void createFontAtlas();
    void createUIPipeline();
    void createGridMesh();

    // --- Texture asset helpers (VulkanAssets.cpp) ---
    void createTextureImage(const std::string& path);
    void createTextureImageView();
    void createTextureSampler();

    // --- Buffer / image utilities ---
    void createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags props,
                     VkImage& image, VkDeviceMemory& mem, uint32_t mipLevels = 1);
    void generateMipmaps(VkImage image, int32_t texW, int32_t texH, uint32_t mipLevels);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer cb);
    void transitionImageLayout(VkImage image, VkFormat fmt,
                               VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t w, uint32_t h);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props, VkBuffer& buffer,
                      VkDeviceMemory& mem, bool deviceAddress = false);
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);

    // --- Rendering (RendererDraw.cpp) ---
    void updateUniformBuffer();
    void renderScene(VkCommandBuffer cb);
    void renderSkinned(VkCommandBuffer cb);
    void renderParticles(VkCommandBuffer cb);
    void renderInGameUI(VkCommandBuffer cb, bool isPlaying);
    void tickParticles(float dt);
    void tickSkinning(float dt);
    std::tuple<std::vector<RtInstance>,
               std::vector<std::string>,
               std::vector<std::pair<VkBuffer, VkBuffer>>> buildRtData();
};
