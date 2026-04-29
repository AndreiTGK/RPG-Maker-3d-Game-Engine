#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include "VulkanAssets.hpp" // Pentru structura Vertex

class EditorOverlay {
public:
    void init(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);
    void cleanup(VkDevice device);
    void drawGrid(VkCommandBuffer commandBuffer, VkPipeline linePipeline, VkPipelineLayout layout);

private:
    MeshResource gridMesh;
    void createGridGeometry(VkDevice device, VkPhysicalDevice physicalDevice, VkCommandPool commandPool, VkQueue graphicsQueue);
};
