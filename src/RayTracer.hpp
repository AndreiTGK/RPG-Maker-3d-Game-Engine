#pragma once

#include <vulkan/vulkan.h>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <map>
#include <string>
#include <algorithm>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Scene.hpp"

static constexpr uint32_t MAX_RT_MESHES = 64;

struct BlasEntry {
    VkAccelerationStructureKHR as  = VK_NULL_HANDLE;
    VkBuffer                   buf = VK_NULL_HANDLE;
    VkDeviceMemory             mem = VK_NULL_HANDLE;
};

struct RtInstance {
    glm::mat4   transform;
    std::string meshName;
    uint32_t    textureIndex;
};

class RayTracer {
public:
    void init(VkDevice logicalDevice, VkPhysicalDevice physDevice);
    bool isReady();

    PFN_vkGetBufferDeviceAddressKHR vkGetBufferDeviceAddressKHR;
    PFN_vkCreateAccelerationStructureKHR vkCreateAccelerationStructureKHR;
    PFN_vkDestroyAccelerationStructureKHR vkDestroyAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR vkGetAccelerationStructureBuildSizesKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR vkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR vkCmdBuildAccelerationStructuresKHR;
    PFN_vkBuildAccelerationStructuresKHR vkBuildAccelerationStructuresKHR;
    PFN_vkCreateRayTracingPipelinesKHR vkCreateRayTracingPipelinesKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR vkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCmdTraceRaysKHR vkCmdTraceRaysKHR;
    VkDeviceAddress getBufferDeviceAddress(VkBuffer buffer);

    std::map<std::string, BlasEntry> blasMap;

    void buildBLAS(const std::string& name, VkBuffer vertexBuffer, uint32_t vertexCount, VkBuffer indexBuffer, uint32_t indexCount, VkCommandPool commandPool, VkQueue graphicsQueue);
    VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
    VkBuffer tlasBuffer = VK_NULL_HANDLE;
    VkDeviceMemory tlasBufferMemory = VK_NULL_HANDLE;
    VkBuffer tlasInstancesBuffer = VK_NULL_HANDLE;
    VkDeviceMemory tlasInstancesBufferMemory = VK_NULL_HANDLE;

    // Persistent scratch buffer — pre-allocated for the current TLAS capacity
    VkBuffer tlasPersistentScratch = VK_NULL_HANDLE;
    VkDeviceMemory tlasPersistentScratchMemory = VK_NULL_HANDLE;

    // How many instances the current TLAS buffers can hold without reallocation
    uint32_t tlasAllocatedInstances = 0;
    uint32_t tlasCurrentInstances   = 0;

    // When true, createDescriptorSets must push updated writes to the GPU
    bool rtDescriptorsDirty = true;
    VkImage storageImage = VK_NULL_HANDLE;
    VkDeviceMemory storageImageMemory = VK_NULL_HANDLE;
    VkImageView storageImageView = VK_NULL_HANDLE;

    void createStorageImage(uint32_t width, uint32_t height);
    void destroyStorageImage();
    void buildTLAS(const std::vector<RtInstance>& instances, const std::vector<std::string>& meshNames, VkCommandPool commandPool, VkQueue graphicsQueue);
    VkTransformMatrixKHR glmMat4ToVkTransformMatrix(const glm::mat4& matrix);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    VkDescriptorSet rtDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSetLayout rtDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool rtDescriptorPool = VK_NULL_HANDLE;

    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets(VkBuffer uniformBuffer, VkDeviceSize uboSize, const std::vector<std::pair<VkBuffer,VkBuffer>>& meshBuffers, const std::vector<VkImageView>& textureImageViews, const std::vector<VkSampler>& textureSamplers);
    void cleanup();

    VkPipelineLayout rtPipelineLayout = VK_NULL_HANDLE;
    VkPipeline rtPipeline = VK_NULL_HANDLE;

    void createRayTracingPipeline();
    std::vector<char> readFile(const std::string& filename);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    VkBuffer sbtBuffer = VK_NULL_HANDLE;
    VkDeviceMemory sbtBufferMemory = VK_NULL_HANDLE;

    VkStridedDeviceAddressRegionKHR raygenRegion{};
    VkStridedDeviceAddressRegionKHR missRegion{};
    VkStridedDeviceAddressRegionKHR hitRegion{};
    VkStridedDeviceAddressRegionKHR callRegion{};

    void createShaderBindingTable();
    uint32_t alignUp(uint32_t size, uint32_t alignment);

private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    bool initialized = false;

    void loadFunctionPointers();
};
