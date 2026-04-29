#include "VulkanEngine.hpp"
#include "EngineLog.hpp"
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_vulkan.h"
#include <stdexcept>
#include <cstring>

void Renderer::init(GLFWwindow* w, Scene* s, Camera* c, Workspace* ws) {
    window = w; scene = s; camera = c; workspace = ws;
    createInstance();
    createSurface();
    setupPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createDepthResources();
    createCommandPool();       // needed early: shadow resources do layout transitions via single-time commands

    createShadowResources();
    createShadowRenderPass();

    createRenderPass();        // HDR scene pass (R16G16B16A16_SFLOAT)
    createHDRResources();      // hdrImage/view/sampler/hdrFramebuffer
    createPostRenderPass();    // fullscreen post → swapchain
    createImguiRenderPass();   // LOAD-mode swapchain pass for RT path
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createShadowPipeline();
    createPostPipeline();      // ACES + bloom pipeline
    createParticlePipeline();  // alpha-blended billboards
    createSkinnedPipeline();   // skeletal animation

    createFramebuffers();      // swapchain framebuffers with postRenderPass
    createFontAtlas();         // embedded 8×8 bitmap font → GPU texture (needs commandPool)
    createUIPipeline();        // in-game UI overlay — needs fontAtlasSampler/View from createFontAtlas

    scanTexturesFolder();

    for (const auto& texName : availableTextures) {
        std::string texturePath = workspace->activeProject.getTexturesPath() + "/" + texName;
        try {
            createTextureImage(texturePath);
            createTextureImageView();
            createTextureSampler();
        } catch (...) {
            LOG_WARNING("Texture load failed at startup: %s", texName.c_str());
        }
    }
    if (availableTextures.empty())
        printf("Avertisment: Nu s-a gasit nicio textura in proiect.\n");

    createFallbackMesh();
    loadAllProjectMeshes();
    createGridMesh();

    if (rayTracer.isReady() && !loadedMeshes.empty()) {
        for (auto& [meshName, mesh] : loadedMeshes) {
            if (mesh.vertexBuffer != VK_NULL_HANDLE && mesh.indexBuffer != VK_NULL_HANDLE)
                rayTracer.buildBLAS(meshName, mesh.vertexBuffer, mesh.vertexCount, mesh.indexBuffer, mesh.indexCount, commandPool, graphicsQueue);
        }
        auto [rtInst, meshNames, meshBuffers] = buildRtData();
        rayTracer.buildTLAS(rtInst, meshNames, commandPool, graphicsQueue);
        rayTracer.createStorageImage(windowExtent.width, windowExtent.height);
        transitionImageLayout(rayTracer.storageImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    }

    createUniformBuffer();

    if (rayTracer.isReady()) {
        rayTracer.createDescriptorSetLayout();
        rayTracer.createDescriptorPool();
        auto [rtInst, meshNames, meshBuffers] = buildRtData();
        rayTracer.createDescriptorSets(uniformBuffer, sizeof(UniformBufferObject), meshBuffers, textureImageViews, textureSamplers);

        rayTracer.createRayTracingPipeline();
        rayTracer.createShaderBindingTable();
    }

    createDescriptorPool();
    createDescriptorSets();
    updateTextureDescriptorSet();
    createCommandBuffer();
    createSyncObjects();
}

void Renderer::initImGui() {
    VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = pool_sizes;
    vkCreateDescriptorPool(device, &pool_info, nullptr, &imguiPool);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = physicalDevice;
    init_info.Device = device;
    init_info.Queue = graphicsQueue;
    init_info.DescriptorPool = imguiPool;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    init_info.PipelineInfoMain.RenderPass = imguiRenderPass;

    ImGui_ImplVulkan_Init(&init_info);
}

void Renderer::cleanup() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    vkDestroyDescriptorPool(device, imguiPool, nullptr);
    vkDestroySemaphore(device, renderFinishedSemaphore, nullptr);
    vkDestroySemaphore(device, imageAvailableSemaphore, nullptr);
    vkDestroyFence(device, inFlightFence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);

    // HDR offscreen resources
    vkDestroyFramebuffer(device, hdrFramebuffer, nullptr);
    vkDestroySampler(device, hdrSampler, nullptr);
    vkDestroyImageView(device, hdrImageView, nullptr);
    vkDestroyImage(device, hdrImage, nullptr);
    vkFreeMemory(device, hdrImageMemory, nullptr);

    // Post-processing pipeline
    vkDestroyPipeline(device, postPipeline, nullptr);
    vkDestroyPipelineLayout(device, postPipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, postDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, postDescriptorSetLayout, nullptr);
    vkDestroyRenderPass(device, postRenderPass, nullptr);
    vkDestroyRenderPass(device, imguiRenderPass, nullptr);

    // Sun shadow atlas
    vkDestroySampler(device, shadowSampler, nullptr);
    vkDestroyImageView(device, shadowImageView, nullptr);
    vkDestroyImage(device, shadowImage, nullptr);
    vkFreeMemory(device, shadowImageMemory, nullptr);
    vkDestroyFramebuffer(device, shadowFramebuffer, nullptr);
    vkDestroyPipeline(device, shadowPipeline, nullptr);
    vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);
    vkDestroyRenderPass(device, shadowRenderPass, nullptr);

    // Cubemap point-light shadows
    vkDestroyPipeline(device, shadowCubePipeline, nullptr);
    vkDestroyPipelineLayout(device, shadowCubePipelineLayout, nullptr);
    vkDestroyRenderPass(device, shadowCubeRenderPass, nullptr);
    for (uint32_t i = 0; i < kMaxCubeLights * 6; i++) {
        vkDestroyFramebuffer(device, shadowCubeFBs[i], nullptr);
        vkDestroyImageView(device, shadowCubeFaceViews[i], nullptr);
    }
    vkDestroySampler(device, shadowCubeSampler, nullptr);
    vkDestroyImageView(device, shadowCubeArrayView, nullptr);
    vkDestroyImage(device, shadowCubeImage, nullptr);
    vkFreeMemory(device, shadowCubeMemory, nullptr);
    vkDestroyImageView(device, shadowCubeDepthView, nullptr);
    vkDestroyImage(device, shadowCubeDepth, nullptr);
    vkFreeMemory(device, shadowCubeDepthMem, nullptr);

    for(size_t i = 0; i < textureImages.size(); i++) {
        vkDestroySampler(device, textureSamplers[i], nullptr);
        vkDestroyImageView(device, textureImageViews[i], nullptr);
        vkDestroyImage(device, textureImages[i], nullptr);
        vkFreeMemory(device, textureImageMemories[i], nullptr);
    }
    textureImages.clear();
    textureImageViews.clear();
    textureSamplers.clear();
    textureImageMemories.clear();

    for (auto fb : swapChainFramebuffers) vkDestroyFramebuffer(device, fb, nullptr);
    for (auto iv : swapChainImageViews) vkDestroyImageView(device, iv, nullptr);
    vkDestroySwapchainKHR(device, swapChain, nullptr);

    for (auto& [name, mesh] : loadedMeshes) {
        vkDestroyBuffer(device, mesh.vertexBuffer, nullptr);
        vkFreeMemory(device, mesh.vertexBufferMemory, nullptr);
        vkDestroyBuffer(device, mesh.indexBuffer, nullptr);
        vkFreeMemory(device, mesh.indexBufferMemory, nullptr);
        for (int i = 0; i < 2; i++) {
            if (mesh.lodVB[i] != VK_NULL_HANDLE) { vkDestroyBuffer(device, mesh.lodVB[i], nullptr); vkFreeMemory(device, mesh.lodVBMem[i], nullptr); }
            if (mesh.lodIB[i] != VK_NULL_HANDLE) { vkDestroyBuffer(device, mesh.lodIB[i], nullptr); vkFreeMemory(device, mesh.lodIBMem[i], nullptr); }
        }
    }
    loadedMeshes.clear();

    vkDestroyBuffer(device, gridMesh.vertexBuffer, nullptr);
    vkFreeMemory(device, gridMesh.vertexBufferMemory, nullptr);
    vkDestroyBuffer(device, gridMesh.indexBuffer, nullptr);
    vkFreeMemory(device, gridMesh.indexBufferMemory, nullptr);

    vkDestroyBuffer(device, uniformBuffer, nullptr);
    vkFreeMemory(device, uniformBufferMemory, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    // Destroy all particle emitter GPU buffers
    for (auto& [id, es] : particleEmitters)
        destroyEmitterState(es);
    particleEmitters.clear();

    // Destroy all skin instances (SSBOs)
    for (auto& [id, si] : skinInstances) {
        if (si.ssbo != VK_NULL_HANDLE) {
            vkUnmapMemory(device, si.mem);
            vkDestroyBuffer(device, si.ssbo, nullptr);
            vkFreeMemory(device, si.mem, nullptr);
        }
    }
    skinInstances.clear();

    // Destroy skinned mesh GPU buffers
    for (auto& [name, smr] : loadedSkinnedMeshes) {
        vkDestroyBuffer(device, smr.vertexBuffer, nullptr);
        vkFreeMemory(device, smr.vertexBufferMemory, nullptr);
        vkDestroyBuffer(device, smr.indexBuffer, nullptr);
        vkFreeMemory(device, smr.indexBufferMemory, nullptr);
    }
    loadedSkinnedMeshes.clear();

    // In-game UI system
    if (uiVertexMapped) { vkUnmapMemory(device, uiVertexBufferMemory); uiVertexMapped = nullptr; }
    vkDestroyBuffer(device, uiVertexBuffer, nullptr);
    vkFreeMemory(device, uiVertexBufferMemory, nullptr);
    vkDestroyDescriptorPool(device, uiDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, uiDescSetLayout, nullptr);
    vkDestroySampler(device, fontAtlasSampler, nullptr);
    vkDestroyImageView(device, fontAtlasView, nullptr);
    vkDestroyImage(device, fontAtlasImage, nullptr);
    vkFreeMemory(device, fontAtlasMemory, nullptr);
    // Also fix skinDesc leak from previous session
    vkDestroyDescriptorPool(device, skinDescPool, nullptr);
    vkDestroyDescriptorSetLayout(device, skinDescSetLayout, nullptr);

    vkDestroyPipeline(device, particlePipeline, nullptr);
    vkDestroyPipelineLayout(device, particlePipelineLayout, nullptr);
    // skinDescPool/skinDescSetLayout survive swapchain recreates (descriptor sets
    // held by skinInstances remain valid). Same for UI pool/layout/font atlas.
    vkDestroyPipeline(device, skinnedPipeline, nullptr);
    vkDestroyPipelineLayout(device, skinnedPipelineLayout, nullptr);
    vkDestroyPipeline(device, uiPipeline, nullptr);
    vkDestroyPipelineLayout(device, uiPipelineLayout, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    rayTracer.cleanup();
    vkDestroyDevice(device, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;
    }
    THROW_ENGINE_ERROR("Failed to find suitable memory type!");
}

std::vector<char> Renderer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) THROW_ENGINE_ERROR("Failed to open file: " + filename);
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0); file.read(buffer.data(), fileSize); file.close();
    return buffer;
}

VkShaderModule Renderer::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size(); ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm; vkCreateShaderModule(device, &ci, nullptr, &sm);
    return sm;
}

void Renderer::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "RPG Maker 3D";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = count;
    ci.ppEnabledExtensionNames = exts;

    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Failed to create instance");
    }
}

void Renderer::createSurface() {
    if(glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        THROW_ENGINE_ERROR("Failed to create surface");
}

void Renderer::setupPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) THROW_ENGINE_ERROR("No Vulkan-capable GPU found.");
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance, &count, devs.data());

    // Prefer discrete GPU over integrated
    physicalDevice = devs[0];
    for (const auto& dev : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(dev, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            physicalDevice = dev;
            break;
        }
    }

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
            if (presentSupport) { graphicsQueueIndex = i; break; }
        }
    }
}

void Renderer::createLogicalDevice() {
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    qci.queueFamilyIndex = graphicsQueueIndex;

    // Enumerate available device extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExts(extCount);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, availableExts.data());

    auto hasExt = [&](const char* name) {
        for (const auto& ext : availableExts)
            if (strcmp(ext.extensionName, name) == 0) return true;
        return false;
    };

    // RT requires three extensions; if any is missing we run in rasterization-only mode
    bool rtSupported = hasExt(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME) &&
                       hasExt(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                       hasExt(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

    std::vector<const char*> devExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    // These four are promoted to Vulkan 1.2 core but listing them is still valid
    if (hasExt(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
        devExts.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    if (hasExt(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME))
        devExts.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    if (hasExt(VK_KHR_SPIRV_1_4_EXTENSION_NAME))
        devExts.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    if (hasExt(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME))
        devExts.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

    if (rtSupported) {
        devExts.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        devExts.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        devExts.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    } else {
        printf("Info: GPU does not support ray tracing extensions — running in rasterization mode.\n");
    }

    VkPhysicalDeviceDescriptorIndexingFeatures indexingFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    indexingFeatures.runtimeDescriptorArray = VK_TRUE;
    indexingFeatures.descriptorBindingPartiallyBound = VK_TRUE;
    indexingFeatures.descriptorBindingVariableDescriptorCount = VK_TRUE;

    VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    bufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
    bufferDeviceAddressFeatures.pNext = &indexingFeatures;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    rayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
    rayTracingPipelineFeatures.pNext = &bufferDeviceAddressFeatures;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    accelerationStructureFeatures.accelerationStructure = VK_TRUE;
    accelerationStructureFeatures.pNext = &rayTracingPipelineFeatures;

    VkPhysicalDeviceFeatures2 physicalDeviceFeatures2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    physicalDeviceFeatures2.features.samplerAnisotropy = VK_TRUE;
    // Only chain RT feature structs when the extensions are actually available
    physicalDeviceFeatures2.pNext = rtSupported
        ? (void*)&accelerationStructureFeatures
        : (void*)&bufferDeviceAddressFeatures;

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = (uint32_t)devExts.size();
    ci.ppEnabledExtensionNames = devExts.data();
    ci.pNext = &physicalDeviceFeatures2;

    if (vkCreateDevice(physicalDevice, &ci, nullptr, &device) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Failed to create logical device.");
    }

    if (rtSupported) {
        rayTracer.init(device, physicalDevice);
    }
    vkGetDeviceQueue(device, graphicsQueueIndex, 0, &graphicsQueue);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    maxSamplerAnisotropy = props.limits.maxSamplerAnisotropy;
}

void Renderer::createSwapChain() {
    int w, h; glfwGetFramebufferSize(window, &w, &h);
    windowExtent = { (uint32_t)w, (uint32_t)h };
    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface; ci.minImageCount = 2; ci.imageFormat = VK_FORMAT_B8G8R8A8_SRGB;
    ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR; ci.imageExtent = windowExtent;
    ci.imageArrayLayers = 1; ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR; ci.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; ci.clipped = VK_TRUE;
    vkCreateSwapchainKHR(device, &ci, nullptr, &swapChain);
}

void Renderer::createImageViews() {
    uint32_t count;
    vkGetSwapchainImagesKHR(device, swapChain, &count, nullptr);

    swapChainImages.resize(count);
    vkGetSwapchainImagesKHR(device, swapChain, &count, swapChainImages.data());

    swapChainImageViews.resize(count);
    for(size_t i = 0; i < count; i++) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = swapChainImages[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_B8G8R8A8_SRGB;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device, &vi, nullptr, &swapChainImageViews[i]);
    }
}

void Renderer::createDepthResources() {
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = windowExtent.width;
    imageInfo.extent.height = windowExtent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkCreateImage(device, &imageInfo, nullptr, &depthImage);

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, depthImage, &memReqs);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkAllocateMemory(device, &allocInfo, nullptr, &depthImageMemory);
    vkBindImageMemory(device, depthImage, depthImageMemory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = depthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    vkCreateImageView(device, &viewInfo, nullptr, &depthImageView);
}

void Renderer::createRenderPass() {
    // HDR offscreen scene pass: renders to R16G16B16A16_SFLOAT.
    // Clear on load; transition to SHADER_READ_ONLY_OPTIMAL so the post pass can sample it.
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};
    VkRenderPassCreateInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;

    vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
}

void Renderer::createFramebuffers() {
    // Swapchain framebuffers are used by the post render pass (color-only, no depth).
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebufferInfo.renderPass = postRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &swapChainImageViews[i];
        framebufferInfo.width = windowExtent.width;
        framebufferInfo.height = windowExtent.height;
        framebufferInfo.layers = 1;
        vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]);
    }
}

void Renderer::createHDRResources() {
    // Create the R16G16B16A16_SFLOAT image that the scene renders into.
    createImage(windowExtent.width, windowExtent.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        hdrImage, hdrImageMemory);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = hdrImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &viewInfo, nullptr, &hdrImageView);

    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &samplerInfo, nullptr, &hdrSampler);

    // Single framebuffer for the HDR scene pass (offscreen, not per-swapchain-image).
    std::array<VkImageView, 2> attachments = {hdrImageView, depthImageView};
    VkFramebufferCreateInfo fbInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbInfo.renderPass = renderPass;  // HDR scene render pass
    fbInfo.attachmentCount = 2;
    fbInfo.pAttachments = attachments.data();
    fbInfo.width = windowExtent.width;
    fbInfo.height = windowExtent.height;
    fbInfo.layers = 1;
    vkCreateFramebuffer(device, &fbInfo, nullptr, &hdrFramebuffer);
}

void Renderer::createPostRenderPass() {
    // Post pass: fullscreen triangle + ImGui → swapchain.
    // DONT_CARE load: the post shader fills every pixel.
    VkAttachmentDescription color{};
    color.format = VK_FORMAT_B8G8R8A8_SRGB;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &color;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    vkCreateRenderPass(device, &rpInfo, nullptr, &postRenderPass);
}

void Renderer::createImguiRenderPass() {
    // Compatible with postRenderPass but preserves existing content (for the RT blit path).
    VkAttachmentDescription color{};
    color.format = VK_FORMAT_B8G8R8A8_SRGB;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    VkRenderPassCreateInfo rpInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpInfo.attachmentCount = 1;
    rpInfo.pAttachments = &color;
    rpInfo.subpassCount = 1;
    rpInfo.pSubpasses = &subpass;
    vkCreateRenderPass(device, &rpInfo, nullptr, &imguiRenderPass);
}

void Renderer::createPostPipeline() {
    // Descriptor set layout: binding 0 = HDR sampler
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslInfo.bindingCount = 1;
    dslInfo.pBindings = &binding;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &postDescriptorSetLayout);

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &postDescriptorPool);

    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = postDescriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &postDescriptorSetLayout;
    vkAllocateDescriptorSets(device, &allocInfo, &postDescriptorSet);

    VkDescriptorImageInfo imgInfo{};
    imgInfo.sampler = hdrSampler;
    imgInfo.imageView = hdrImageView;
    imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = postDescriptorSet;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &imgInfo;
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(PostSettings); // 12 bytes: 3 floats

    VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &postDescriptorSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &postPipelineLayout);

    auto vertCode = readFile("shaders/post.vert.spv");
    auto fragCode = readFile("shaders/post.frag.spv");
    VkShaderModule vertMod = createShaderModule(vertCode);
    VkShaderModule fragMod = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertMod;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragMod;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &blendAttachment;

    std::vector<VkDynamicState> dynamics = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamics.size());
    dynamicState.pDynamicStates = dynamics.data();

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = postPipelineLayout;
    pipelineInfo.renderPass = postRenderPass;
    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &postPipeline);

    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
}

// --- NOU: Functia pentru crearea pipeline-ului offscreen de umbre ---
void Renderer::createShadowPipeline() {
    auto vertShaderCode = readFile("shaders/shadow.vert.spv");
    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShaderModule;
    vertStage.pName = "main";

    auto bindingDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
    vertexInput.pVertexAttributeDescriptions = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_FRONT_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_TRUE;

    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    std::vector<VkDynamicState> dynamics = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamics.size());
    dynamicState.pDynamicStates = dynamics.data();

    VkPushConstantRange shadowPCRange{};
    shadowPCRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    shadowPCRange.offset = 0;
    shadowPCRange.size = sizeof(ShadowPushConstant); // 128 bytes: modelMatrix + lightSpaceMatrix

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &shadowPCRange;

    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &shadowPipelineLayout);

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 1;
    pipelineInfo.pStages = &vertStage;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = shadowPipelineLayout;
    pipelineInfo.renderPass = shadowRenderPass;

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);

    // ── Cubemap shadow pipeline (vert + frag, writes R32_SFLOAT linear depth) ─
    {
        auto cubeVertCode = readFile("shaders/shadowcube.vert.spv");
        auto cubeFragCode = readFile("shaders/shadowcube.frag.spv");
        VkShaderModule cubeVert = createShaderModule(cubeVertCode);
        VkShaderModule cubeFrag = createShaderModule(cubeFragCode);

        VkPipelineShaderStageCreateInfo cubeStages[2]{};
        cubeStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cubeStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        cubeStages[0].module = cubeVert;
        cubeStages[0].pName  = "main";
        cubeStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cubeStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        cubeStages[1].module = cubeFrag;
        cubeStages[1].pName  = "main";

        auto bDesc = Vertex::getBindingDescription();
        auto aDesc = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &bDesc;
        vi.vertexAttributeDescriptionCount = static_cast<uint32_t>(aDesc.size());
        vi.pVertexAttributeDescriptions    = aDesc.data();

        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vs.viewportCount = 1;
        vs.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.lineWidth   = 1.0f;
        rs.cullMode    = VK_CULL_MODE_NONE; // no culling — avoids Peter Panning on any face

        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments    = &cba;

        std::vector<VkDynamicState> dynStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynState.dynamicStateCount = static_cast<uint32_t>(dynStates.size());
        dynState.pDynamicStates    = dynStates.data();

        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.offset     = 0;
        pcr.size       = sizeof(CubeShadowPushConstant); // 144 bytes

        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges    = &pcr;
        vkCreatePipelineLayout(device, &pli, nullptr, &shadowCubePipelineLayout);

        VkGraphicsPipelineCreateInfo gpi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gpi.stageCount          = 2;
        gpi.pStages             = cubeStages;
        gpi.pVertexInputState   = &vi;
        gpi.pInputAssemblyState = &ia;
        gpi.pViewportState      = &vs;
        gpi.pRasterizationState = &rs;
        gpi.pMultisampleState   = &ms;
        gpi.pDepthStencilState  = &ds;
        gpi.pColorBlendState    = &cb;
        gpi.pDynamicState       = &dynState;
        gpi.layout              = shadowCubePipelineLayout;
        gpi.renderPass          = shadowCubeRenderPass;
        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpi, nullptr, &shadowCubePipeline);

        vkDestroyShaderModule(device, cubeVert, nullptr);
        vkDestroyShaderModule(device, cubeFrag, nullptr);
    }
}

void Renderer::createGraphicsPipeline() {
    auto vertShaderCode = readFile("shaders/vert.spv");
    auto fragShaderCode = readFile("shaders/frag.spv");

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(SimplePushConstantData);

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertShaderModule;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragShaderModule;
    fragStage.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertStage, fragStage};

    auto bindingDesc = Vertex::getBindingDescription();
    auto attrDesc = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDesc.size());
    vertexInput.pVertexAttributeDescriptions = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector<VkDynamicState> dynamics = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamics.size());
    dynamicState.pDynamicStates = dynamics.data();

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare: Nu am putut crea pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare: Nu am putut crea graphics pipeline!");
    }

    VkPipelineInputAssemblyStateCreateInfo lineAssembly{};
    lineAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    lineAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    lineAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineRasterizationStateCreateInfo lineRasterizer{};
    lineRasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    lineRasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    lineRasterizer.lineWidth = 1.0f;
    lineRasterizer.cullMode = VK_CULL_MODE_NONE;
    lineRasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineDepthStencilStateCreateInfo lineDepth{};
    lineDepth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    lineDepth.depthTestEnable = VK_TRUE;
    lineDepth.depthWriteEnable = VK_TRUE;
    lineDepth.depthCompareOp = VK_COMPARE_OP_LESS;

    VkGraphicsPipelineCreateInfo linePipelineInfo = pipelineInfo;
    linePipelineInfo.pInputAssemblyState = &lineAssembly;
    linePipelineInfo.pRasterizationState = &lineRasterizer;
    linePipelineInfo.pDepthStencilState = &lineDepth;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &linePipelineInfo, nullptr, &linePipeline) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare: Nu am putut crea pipeline-ul pentru linii!");
    }

    vkDestroyShaderModule(device, fragShaderModule, nullptr);
    vkDestroyShaderModule(device, vertShaderModule, nullptr);
}

void Renderer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding    {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1,   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutBinding samplerLayoutBinding{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutBinding shadowMapBinding    {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutBinding shadowCubeBinding  {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,   VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};

    std::array<VkDescriptorSetLayoutBinding, 4> bindings = {uboLayoutBinding, samplerLayoutBinding, shadowMapBinding, shadowCubeBinding};

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    VkDescriptorBindingFlags bindFlags[4] = {0, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, 0, 0};
    flagsInfo.bindingCount = 4;
    flagsInfo.pBindingFlags = bindFlags;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.pNext = &flagsInfo;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout);
}

void Renderer::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 102; // 100 textures + sun shadow + cube shadow

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);
}

void Renderer::createDescriptorSets() {
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;
    vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet);

    VkDescriptorBufferInfo bufferInfo{uniformBuffer, 0, sizeof(UniformBufferObject)};

    std::vector<VkDescriptorImageInfo> texImageInfos(textureImageViews.size());
    for (size_t i = 0; i < textureImageViews.size(); ++i) {
        texImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        texImageInfos[i].imageView = textureImageViews[i];
        texImageInfos[i].sampler = textureSamplers[i];
    }

    VkDescriptorImageInfo shadowMapInfo{};
    shadowMapInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    shadowMapInfo.imageView = shadowImageView;
    shadowMapInfo.sampler = shadowSampler;

    VkWriteDescriptorSet uboWrite{};
    uboWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uboWrite.dstSet = descriptorSet;
    uboWrite.dstBinding = 0;
    uboWrite.dstArrayElement = 0;
    uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboWrite.descriptorCount = 1;
    uboWrite.pBufferInfo = &bufferInfo;

    VkWriteDescriptorSet textureWrite{};
    textureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    textureWrite.dstSet = descriptorSet;
    textureWrite.dstBinding = 1;
    textureWrite.dstArrayElement = 0;
    textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureWrite.descriptorCount = static_cast<uint32_t>(texImageInfos.size());
    textureWrite.pImageInfo = texImageInfos.data();

    VkWriteDescriptorSet shadowWrite{};
    shadowWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    shadowWrite.dstSet = descriptorSet;
    shadowWrite.dstBinding = 2;
    shadowWrite.dstArrayElement = 0;
    shadowWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowWrite.descriptorCount = 1;
    shadowWrite.pImageInfo = &shadowMapInfo;

    VkDescriptorImageInfo shadowCubeInfo{};
    shadowCubeInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    shadowCubeInfo.imageView   = shadowCubeArrayView;
    shadowCubeInfo.sampler     = shadowCubeSampler;

    VkWriteDescriptorSet shadowCubeWrite{};
    shadowCubeWrite.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    shadowCubeWrite.dstSet          = descriptorSet;
    shadowCubeWrite.dstBinding      = 3;
    shadowCubeWrite.dstArrayElement = 0;
    shadowCubeWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowCubeWrite.descriptorCount = 1;
    shadowCubeWrite.pImageInfo      = &shadowCubeInfo;

    std::vector<VkWriteDescriptorSet> descriptorWrites = { uboWrite };
    if (!texImageInfos.empty()) descriptorWrites.push_back(textureWrite);
    if (shadowImageView    != VK_NULL_HANDLE) descriptorWrites.push_back(shadowWrite);
    if (shadowCubeArrayView != VK_NULL_HANDLE) descriptorWrites.push_back(shadowCubeWrite);
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void Renderer::createUniformBuffer() {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bufferSize; bi.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bi, nullptr, &uniformBuffer);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, uniformBuffer, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size; ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(device, &ai, nullptr, &uniformBufferMemory);
    vkBindBufferMemory(device, uniformBuffer, uniformBufferMemory, 0);
    vkMapMemory(device, uniformBufferMemory, 0, bufferSize, 0, &uniformBufferMapped);
}

void Renderer::createCommandPool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = graphicsQueueIndex;
    vkCreateCommandPool(device, &ci, nullptr, &commandPool);
}

void Renderer::createCommandBuffer() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &ai, &commandBuffer);
}

void Renderer::createSyncObjects() {
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    vkCreateSemaphore(device, &si, nullptr, &imageAvailableSemaphore);
    vkCreateSemaphore(device, &si, nullptr, &renderFinishedSemaphore);
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT; // Start signaled so the first frame doesn't deadlock
    vkCreateFence(device, &fi, nullptr, &inFlightFence);
}

void Renderer::updateTextureDescriptorSet() {
    if(textureImageViews.empty()) return;

    vkDeviceWaitIdle(device);

    std::vector<VkDescriptorImageInfo> texImageInfos(textureImageViews.size());
    for (size_t i = 0; i < textureImageViews.size(); ++i) {
        texImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        texImageInfos[i].imageView = textureImageViews[i];
        texImageInfos[i].sampler = textureSamplers[i];
    }

    VkWriteDescriptorSet textureWrite{};
    textureWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    textureWrite.dstSet = descriptorSet;
    textureWrite.dstBinding = 1;
    textureWrite.dstArrayElement = 0;
    textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureWrite.descriptorCount = static_cast<uint32_t>(texImageInfos.size());
    textureWrite.pImageInfo = texImageInfos.data();

    vkUpdateDescriptorSets(device, 1, &textureWrite, 0, nullptr);
}

void Renderer::createParticlePipeline() {
    auto vertCode = readFile("shaders/particle.vert.spv");
    auto fragCode = readFile("shaders/particle.frag.spv");
    VkShaderModule vertMod = createShaderModule(vertCode);
    VkShaderModule fragMod = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT; vertStage.module = vertMod; vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT; fragStage.module = fragMod; fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // Vertex input: one binding, attributes for position/uv/color
    VkVertexInputBindingDescription binding{};
    binding.binding = 0; binding.stride = sizeof(ParticleVertex); binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,  offsetof(ParticleVertex, position)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT,     offsetof(ParticleVertex, uv)};
    attrs[2] = {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(ParticleVertex, color)};

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount = 1; vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 3; vertexInput.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1; vs.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL; rast.lineWidth = 1.0f; rast.cullMode = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_FALSE; // transparent: test but don't write
    ds.depthCompareOp = VK_COMPARE_OP_LESS;

    // Alpha blend
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &blend;

    std::vector<VkDynamicState> dynamics = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynamics.data();

    VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pl.setLayoutCount = 1; pl.pSetLayouts = &descriptorSetLayout;
    vkCreatePipelineLayout(device, &pl, nullptr, &particlePipelineLayout);

    VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vertexInput; pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vs; pi.pRasterizationState = &rast;
    pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dyn;
    pi.layout = particlePipelineLayout;
    pi.renderPass = renderPass; // HDR scene pass

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &particlePipeline);
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
}

void Renderer::createSkinnedPipeline() {
    // skinDescSetLayout: set=1, binding=0 — SSBO with joint matrices (vertex stage)
    VkDescriptorSetLayoutBinding ssboBinding{};
    ssboBinding.binding         = 0;
    ssboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboBinding.descriptorCount = 1;
    ssboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslInfo.bindingCount = 1;
    dslInfo.pBindings    = &ssboBinding;
    vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &skinDescSetLayout);

    // skinDescPool: up to 32 sets (one per skinned entity)
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets       = 32;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &skinDescPool);

    // skinnedPipelineLayout: set0 = main descriptorSetLayout, set1 = skinDescSetLayout
    VkDescriptorSetLayout setLayouts[] = { descriptorSetLayout, skinDescSetLayout };

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(SimplePushConstantData);

    VkPipelineLayoutCreateInfo plInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount         = 2;
    plInfo.pSetLayouts            = setLayouts;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges    = &pcRange;
    vkCreatePipelineLayout(device, &plInfo, nullptr, &skinnedPipelineLayout);

    auto vertCode = readFile("shaders/skinned.vert.spv");
    auto fragCode = readFile("shaders/frag.spv");  // reuse the same fragment shader
    VkShaderModule vertMod = createShaderModule(vertCode);
    VkShaderModule fragMod = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT; vertStage.module = vertMod; vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT; fragStage.module = fragMod; fragStage.pName = "main";
    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    auto bindingDesc = SkinnedVertex::getBindingDescription();
    auto attrDesc    = SkinnedVertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = (uint32_t)attrDesc.size();
    vertexInput.pVertexAttributeDescriptions    = attrDesc.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1; vs.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL; rast.lineWidth = 1.0f;
    rast.cullMode    = VK_CULL_MODE_NONE; rast.frontFace = VK_FRONT_FACE_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAtt{};
    blendAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1; cb.pAttachments = &blendAtt;

    std::vector<VkDynamicState> dynamics = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2; dyn.pDynamicStates = dynamics.data();

    VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pci.stageCount          = 2; pci.pStages = stages;
    pci.pVertexInputState   = &vertexInput; pci.pInputAssemblyState = &ia;
    pci.pViewportState      = &vs; pci.pRasterizationState = &rast;
    pci.pMultisampleState   = &ms; pci.pDepthStencilState  = &ds;
    pci.pColorBlendState    = &cb; pci.pDynamicState       = &dyn;
    pci.layout              = skinnedPipelineLayout;
    pci.renderPass          = renderPass;  // HDR scene pass

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pci, nullptr, &skinnedPipeline);
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
}

void Renderer::cleanupSwapChain() {
    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);
    if (rayTracer.isReady()) {
        rayTracer.destroyStorageImage();
    }
    for (auto framebuffer : swapChainFramebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }

    // HDR offscreen resources (size-dependent)
    vkDestroyFramebuffer(device, hdrFramebuffer, nullptr);
    vkDestroySampler(device, hdrSampler, nullptr);
    vkDestroyImageView(device, hdrImageView, nullptr);
    vkDestroyImage(device, hdrImage, nullptr);
    vkFreeMemory(device, hdrImageMemory, nullptr);

    // Post-processing pipeline (references hdrImageView + postRenderPass)
    vkDestroyPipeline(device, postPipeline, nullptr);
    vkDestroyPipelineLayout(device, postPipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, postDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, postDescriptorSetLayout, nullptr);
    vkDestroyRenderPass(device, postRenderPass, nullptr);
    vkDestroyRenderPass(device, imguiRenderPass, nullptr);

    vkDestroyPipeline(device, particlePipeline, nullptr);
    vkDestroyPipelineLayout(device, particlePipelineLayout, nullptr);
    vkDestroyPipeline(device, linePipeline, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    for (auto imageView : swapChainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }

    vkDestroySwapchainKHR(device, swapChain, nullptr);
}

void Renderer::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createDepthResources();
    createRenderPass();        // HDR scene pass
    createHDRResources();      // hdrImage/view/sampler/framebuffer
    createPostRenderPass();    // post → swapchain pass
    createImguiRenderPass();   // LOAD-mode pass for RT path ImGui
    createGraphicsPipeline();
    createPostPipeline();       // fullscreen ACES + bloom
    createParticlePipeline();   // alpha-blended billboards
    createSkinnedPipeline();    // skeletal animation
    createUIPipeline();         // in-game UI overlay
    createFramebuffers();       // swapchain framebuffers (postRenderPass)

    if (rayTracer.isReady()) {
        rayTracer.createStorageImage(windowExtent.width, windowExtent.height);
        transitionImageLayout(rayTracer.storageImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        auto [rtInst, meshNames, meshBuffers] = buildRtData();
        rayTracer.createDescriptorSets(uniformBuffer, sizeof(UniformBufferObject), meshBuffers, textureImageViews, textureSamplers);
    }
}

void Renderer::createShadowResources() {
    // ── Sun shadow atlas (2D, D32_SFLOAT, 4096×4096, tile 0 = 2048×2048) ─────
    {
        VkFormat fmt = VK_FORMAT_D32_SFLOAT;
        createImage(4096, 4096, fmt, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, shadowImage, shadowImageMemory);

        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = shadowImage;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCreateImageView(device, &vi, nullptr, &shadowImageView);

        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter     = VK_FILTER_LINEAR;
        si.minFilter     = VK_FILTER_LINEAR;
        si.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.maxAnisotropy = 1.0f;
        si.borderColor   = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        si.maxLod        = 1.0f;
        vkCreateSampler(device, &si, nullptr, &shadowSampler);
    }

    // ── Omnidirectional point-light cubemap array (R32_SFLOAT, 512×512, 24 layers) ──
    {
        const uint32_t layers = kMaxCubeLights * 6;

        // Color image: stores linear depth [0,1] per face
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType   = VK_IMAGE_TYPE_2D;
        ci.format      = VK_FORMAT_R32_SFLOAT;
        ci.extent      = {kCubeShadowRes, kCubeShadowRes, 1};
        ci.mipLevels   = 1;
        ci.arrayLayers = layers;
        ci.samples     = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ci.usage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ci.flags       = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        vkCreateImage(device, &ci, nullptr, &shadowCubeImage);

        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(device, shadowCubeImage, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize  = mr.size;
        ai.memoryTypeIndex = findMemoryType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &ai, nullptr, &shadowCubeMemory);
        vkBindImageMemory(device, shadowCubeImage, shadowCubeMemory, 0);

        // Cube-array view for sampling in the main shader
        VkImageViewCreateInfo av{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        av.image    = shadowCubeImage;
        av.viewType = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
        av.format   = VK_FORMAT_R32_SFLOAT;
        av.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
        vkCreateImageView(device, &av, nullptr, &shadowCubeArrayView);

        // Per-face views for framebuffer attachments
        for (uint32_t i = 0; i < layers; i++) {
            VkImageViewCreateInfo fv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            fv.image    = shadowCubeImage;
            fv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            fv.format   = VK_FORMAT_R32_SFLOAT;
            fv.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, i, 1};
            vkCreateImageView(device, &fv, nullptr, &shadowCubeFaceViews[i]);
        }

        // Shared depth buffer for Z-testing during cube face renders (not stored)
        VkImageCreateInfo dc{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        dc.imageType   = VK_IMAGE_TYPE_2D;
        dc.format      = VK_FORMAT_D32_SFLOAT;
        dc.extent      = {kCubeShadowRes, kCubeShadowRes, 1};
        dc.mipLevels   = 1;
        dc.arrayLayers = 1;
        dc.samples     = VK_SAMPLE_COUNT_1_BIT;
        dc.tiling      = VK_IMAGE_TILING_OPTIMAL;
        dc.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        dc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateImage(device, &dc, nullptr, &shadowCubeDepth);

        VkMemoryRequirements dmr;
        vkGetImageMemoryRequirements(device, shadowCubeDepth, &dmr);
        VkMemoryAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        dai.allocationSize  = dmr.size;
        dai.memoryTypeIndex = findMemoryType(dmr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(device, &dai, nullptr, &shadowCubeDepthMem);
        vkBindImageMemory(device, shadowCubeDepth, shadowCubeDepthMem, 0);

        VkImageViewCreateInfo dv{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        dv.image    = shadowCubeDepth;
        dv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        dv.format   = VK_FORMAT_D32_SFLOAT;
        dv.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCreateImageView(device, &dv, nullptr, &shadowCubeDepthView);

        // Sampler for the cubemap array
        VkSamplerCreateInfo cs{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        cs.magFilter     = VK_FILTER_LINEAR;
        cs.minFilter     = VK_FILTER_LINEAR;
        cs.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        cs.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        cs.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        cs.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        cs.maxAnisotropy = 1.0f;
        cs.maxLod        = 1.0f;
        vkCreateSampler(device, &cs, nullptr, &shadowCubeSampler);

        // Transition cubemap to SHADER_READ_ONLY so unused faces read 1.0 (no shadow)
        VkCommandBuffer cb = beginSingleTimeCommands();
        VkImageMemoryBarrier initBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        initBarrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        initBarrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        initBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        initBarrier.image               = shadowCubeImage;
        initBarrier.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
        initBarrier.srcAccessMask       = 0;
        initBarrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &initBarrier);
        endSingleTimeCommands(cb);
    }
}

void Renderer::createShadowRenderPass() {
    VkAttachmentDescription attachmentDescription{};
    attachmentDescription.format = VK_FORMAT_D32_SFLOAT;
    attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthReference{};
    depthReference.attachment = 0;
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0;
    subpass.pDepthStencilAttachment = &depthReference;

    std::array<VkSubpassDependency, 2> dependencies;
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attachmentDescription;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    vkCreateRenderPass(device, &renderPassInfo, nullptr, &shadowRenderPass);

    VkFramebufferCreateInfo framebufferInfo{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebufferInfo.renderPass = shadowRenderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &shadowImageView;
    framebufferInfo.width = 4096;
    framebufferInfo.height = 4096;
    framebufferInfo.layers = 1;
    vkCreateFramebuffer(device, &framebufferInfo, nullptr, &shadowFramebuffer);

    // ── Cubemap shadow render pass ─────────────────────────────────────────────
    // Color attachment: R32_SFLOAT, stores linear depth; SHADER_READ_ONLY ↔ attachment
    VkAttachmentDescription colorAtt{};
    colorAtt.format         = VK_FORMAT_R32_SFLOAT;
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    // Depth attachment: D32_SFLOAT, shared temp buffer, not stored
    VkAttachmentDescription depthAtt{};
    depthAtt.format         = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAtt.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription cubeSubpass{};
    cubeSubpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    cubeSubpass.colorAttachmentCount    = 1;
    cubeSubpass.pColorAttachments       = &colorRef;
    cubeSubpass.pDepthStencilAttachment = &depthRef;

    std::array<VkSubpassDependency, 2> cubeDeps;
    cubeDeps[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    cubeDeps[0].dstSubpass      = 0;
    cubeDeps[0].srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    cubeDeps[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    cubeDeps[0].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    cubeDeps[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    cubeDeps[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    cubeDeps[1].srcSubpass      = 0;
    cubeDeps[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    cubeDeps[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    cubeDeps[1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    cubeDeps[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    cubeDeps[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    cubeDeps[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    std::array<VkAttachmentDescription, 2> cubeAtts = {colorAtt, depthAtt};
    VkRenderPassCreateInfo cubeRPI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    cubeRPI.attachmentCount = 2;
    cubeRPI.pAttachments    = cubeAtts.data();
    cubeRPI.subpassCount    = 1;
    cubeRPI.pSubpasses      = &cubeSubpass;
    cubeRPI.dependencyCount = 2;
    cubeRPI.pDependencies   = cubeDeps.data();
    vkCreateRenderPass(device, &cubeRPI, nullptr, &shadowCubeRenderPass);

    // Initialize shared depth buffer layout
    {
        VkCommandBuffer cb = beginSingleTimeCommands();
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = shadowCubeDepth;
        b.subresourceRange    = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
        endSingleTimeCommands(cb);
    }

    // Per-face framebuffers
    for (uint32_t i = 0; i < kMaxCubeLights * 6; i++) {
        std::array<VkImageView, 2> atts = {shadowCubeFaceViews[i], shadowCubeDepthView};
        VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbi.renderPass      = shadowCubeRenderPass;
        fbi.attachmentCount = 2;
        fbi.pAttachments    = atts.data();
        fbi.width           = kCubeShadowRes;
        fbi.height          = kCubeShadowRes;
        fbi.layers          = 1;
        vkCreateFramebuffer(device, &fbi, nullptr, &shadowCubeFBs[i]);
    }
}

// ---------------------------------------------------------------------------
// In-game UI: font atlas (128×64, R8_UNORM, 16 chars/row × 8 rows, each 8×8 px)
// Public-domain IBM PC 8×8 OEM font — 96 printable ASCII chars (32–127)
// Each entry: 8 bytes, one per pixel row, MSB = leftmost pixel
// ---------------------------------------------------------------------------
static const uint8_t kFontData[96][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 32 space
    {0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00}, // 33 !
    {0x36,0x36,0x00,0x00,0x00,0x00,0x00,0x00}, // 34 "
    {0x36,0x36,0x7F,0x36,0x7F,0x36,0x36,0x00}, // 35 #
    {0x0C,0x3E,0x03,0x1E,0x30,0x1F,0x0C,0x00}, // 36 $
    {0x00,0x63,0x33,0x18,0x0C,0x66,0x63,0x00}, // 37 %
    {0x1C,0x36,0x1C,0x6E,0x3B,0x33,0x6E,0x00}, // 38 &
    {0x06,0x06,0x03,0x00,0x00,0x00,0x00,0x00}, // 39 '
    {0x18,0x0C,0x06,0x06,0x06,0x0C,0x18,0x00}, // 40 (
    {0x06,0x0C,0x18,0x18,0x18,0x0C,0x06,0x00}, // 41 )
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, // 42 *
    {0x00,0x0C,0x0C,0x3F,0x0C,0x0C,0x00,0x00}, // 43 +
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x06}, // 44 ,
    {0x00,0x00,0x00,0x3F,0x00,0x00,0x00,0x00}, // 45 -
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C,0x00}, // 46 .
    {0x60,0x30,0x18,0x0C,0x06,0x03,0x01,0x00}, // 47 /
    {0x3E,0x63,0x73,0x7B,0x6F,0x67,0x3E,0x00}, // 48 0
    {0x0C,0x0E,0x0C,0x0C,0x0C,0x0C,0x3F,0x00}, // 49 1
    {0x1E,0x33,0x30,0x1C,0x06,0x33,0x3F,0x00}, // 50 2
    {0x1E,0x33,0x30,0x1C,0x30,0x33,0x1E,0x00}, // 51 3
    {0x38,0x3C,0x36,0x33,0x7F,0x30,0x78,0x00}, // 52 4
    {0x3F,0x03,0x1F,0x30,0x30,0x33,0x1E,0x00}, // 53 5
    {0x1C,0x06,0x03,0x1F,0x33,0x33,0x1E,0x00}, // 54 6
    {0x3F,0x33,0x30,0x18,0x0C,0x0C,0x0C,0x00}, // 55 7
    {0x1E,0x33,0x33,0x1E,0x33,0x33,0x1E,0x00}, // 56 8
    {0x1E,0x33,0x33,0x3E,0x30,0x18,0x0E,0x00}, // 57 9
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x00}, // 58 :
    {0x00,0x0C,0x0C,0x00,0x00,0x0C,0x0C,0x06}, // 59 ;
    {0x18,0x0C,0x06,0x03,0x06,0x0C,0x18,0x00}, // 60 <
    {0x00,0x00,0x3F,0x00,0x00,0x3F,0x00,0x00}, // 61 =
    {0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00}, // 62 >
    {0x1E,0x33,0x30,0x18,0x0C,0x00,0x0C,0x00}, // 63 ?
    {0x3E,0x63,0x7B,0x7B,0x7B,0x03,0x1E,0x00}, // 64 @
    {0x0C,0x1E,0x33,0x33,0x3F,0x33,0x33,0x00}, // 65 A
    {0x3F,0x66,0x66,0x3E,0x66,0x66,0x3F,0x00}, // 66 B
    {0x3C,0x66,0x03,0x03,0x03,0x66,0x3C,0x00}, // 67 C
    {0x1F,0x36,0x66,0x66,0x66,0x36,0x1F,0x00}, // 68 D
    {0x7F,0x46,0x16,0x1E,0x16,0x46,0x7F,0x00}, // 69 E
    {0x7F,0x46,0x16,0x1E,0x16,0x06,0x0F,0x00}, // 70 F
    {0x3C,0x66,0x03,0x03,0x73,0x66,0x7C,0x00}, // 71 G
    {0x33,0x33,0x33,0x3F,0x33,0x33,0x33,0x00}, // 72 H
    {0x1E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // 73 I
    {0x78,0x30,0x30,0x30,0x33,0x33,0x1E,0x00}, // 74 J
    {0x67,0x66,0x36,0x1E,0x36,0x66,0x67,0x00}, // 75 K
    {0x0F,0x06,0x06,0x06,0x46,0x66,0x7F,0x00}, // 76 L
    {0x63,0x77,0x7F,0x7F,0x6B,0x63,0x63,0x00}, // 77 M
    {0x63,0x67,0x6F,0x7B,0x73,0x63,0x63,0x00}, // 78 N
    {0x1C,0x36,0x63,0x63,0x63,0x36,0x1C,0x00}, // 79 O
    {0x3F,0x66,0x66,0x3E,0x06,0x06,0x0F,0x00}, // 80 P
    {0x1E,0x33,0x33,0x33,0x3B,0x1E,0x38,0x00}, // 81 Q
    {0x3F,0x66,0x66,0x3E,0x36,0x66,0x67,0x00}, // 82 R
    {0x1E,0x33,0x07,0x0E,0x38,0x33,0x1E,0x00}, // 83 S
    {0x3F,0x2D,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // 84 T
    {0x33,0x33,0x33,0x33,0x33,0x33,0x3F,0x00}, // 85 U
    {0x33,0x33,0x33,0x33,0x33,0x1E,0x0C,0x00}, // 86 V
    {0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00}, // 87 W
    {0x63,0x63,0x36,0x1C,0x1C,0x36,0x63,0x00}, // 88 X
    {0x33,0x33,0x33,0x1E,0x0C,0x0C,0x1E,0x00}, // 89 Y
    {0x7F,0x63,0x31,0x18,0x4C,0x66,0x7F,0x00}, // 90 Z
    {0x1E,0x06,0x06,0x06,0x06,0x06,0x1E,0x00}, // 91 [
    {0x03,0x06,0x0C,0x18,0x30,0x60,0x40,0x00}, // 92 backslash
    {0x1E,0x18,0x18,0x18,0x18,0x18,0x1E,0x00}, // 93 ]
    {0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00}, // 94 ^
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF}, // 95 _
    {0x0C,0x0C,0x18,0x00,0x00,0x00,0x00,0x00}, // 96 `
    {0x00,0x00,0x1E,0x30,0x3E,0x33,0x6E,0x00}, // 97  a
    {0x07,0x06,0x06,0x3E,0x66,0x66,0x3B,0x00}, // 98  b
    {0x00,0x00,0x1E,0x33,0x03,0x33,0x1E,0x00}, // 99  c
    {0x38,0x30,0x30,0x3E,0x33,0x33,0x6E,0x00}, // 100 d
    {0x00,0x00,0x1E,0x33,0x3F,0x03,0x1E,0x00}, // 101 e
    {0x1C,0x36,0x06,0x0F,0x06,0x06,0x0F,0x00}, // 102 f
    {0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x1F}, // 103 g
    {0x07,0x06,0x36,0x6E,0x66,0x66,0x67,0x00}, // 104 h
    {0x0C,0x00,0x0E,0x0C,0x0C,0x0C,0x1E,0x00}, // 105 i
    {0x30,0x00,0x30,0x30,0x30,0x33,0x33,0x1E}, // 106 j
    {0x07,0x06,0x66,0x36,0x1E,0x36,0x67,0x00}, // 107 k
    {0x0E,0x0C,0x0C,0x0C,0x0C,0x0C,0x1E,0x00}, // 108 l
    {0x00,0x00,0x33,0x7F,0x7F,0x6B,0x63,0x00}, // 109 m
    {0x00,0x00,0x1F,0x33,0x33,0x33,0x33,0x00}, // 110 n
    {0x00,0x00,0x1E,0x33,0x33,0x33,0x1E,0x00}, // 111 o
    {0x00,0x00,0x3B,0x66,0x66,0x3E,0x06,0x0F}, // 112 p
    {0x00,0x00,0x6E,0x33,0x33,0x3E,0x30,0x78}, // 113 q
    {0x00,0x00,0x3B,0x6E,0x66,0x06,0x0F,0x00}, // 114 r
    {0x00,0x00,0x3E,0x03,0x1E,0x30,0x1F,0x00}, // 115 s
    {0x08,0x0C,0x3E,0x0C,0x0C,0x2C,0x18,0x00}, // 116 t
    {0x00,0x00,0x33,0x33,0x33,0x33,0x6E,0x00}, // 117 u
    {0x00,0x00,0x33,0x33,0x33,0x1E,0x0C,0x00}, // 118 v
    {0x00,0x00,0x63,0x6B,0x7F,0x7F,0x36,0x00}, // 119 w
    {0x00,0x00,0x63,0x36,0x1C,0x36,0x63,0x00}, // 120 x
    {0x00,0x00,0x33,0x33,0x33,0x3E,0x30,0x1F}, // 121 y
    {0x00,0x00,0x3F,0x19,0x0C,0x26,0x3F,0x00}, // 122 z
    {0x38,0x0C,0x0C,0x07,0x0C,0x0C,0x38,0x00}, // 123 {
    {0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00}, // 124 |
    {0x07,0x0C,0x0C,0x38,0x0C,0x0C,0x07,0x00}, // 125 }
    {0x6E,0x3B,0x00,0x00,0x00,0x00,0x00,0x00}, // 126 ~
    {0xFF,0x81,0x81,0x81,0x81,0x81,0x81,0xFF}, // 127 DEL (box)
};

void Renderer::createFontAtlas() {
    // Build 128×64 R8 pixel data
    constexpr int kAtlasW = 128, kAtlasH = 64;
    std::array<uint8_t, kAtlasW * kAtlasH> pixels = {};

    for (int c = 32; c < 128; c++) {
        int col    = c % 16;
        int row    = c / 16;
        int startX = col * 8;
        int startY = row * 8;
        for (int py = 0; py < 8; py++) {
            uint8_t bits = kFontData[c - 32][py];
            for (int px = 0; px < 8; px++) {
                bool set = (bits >> (7 - px)) & 1;
                pixels[(startY + py) * kAtlasW + (startX + px)] = set ? 255u : 0u;
            }
        }
    }

    VkDeviceSize imgSize = kAtlasW * kAtlasH;
    VkBuffer stagingBuf; VkDeviceMemory stagingMem;
    createBuffer(imgSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuf, stagingMem);
    void* data;
    vkMapMemory(device, stagingMem, 0, imgSize, 0, &data);
    memcpy(data, pixels.data(), imgSize);
    vkUnmapMemory(device, stagingMem);

    createImage(kAtlasW, kAtlasH, VK_FORMAT_R8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        fontAtlasImage, fontAtlasMemory);

    transitionImageLayout(fontAtlasImage, VK_FORMAT_R8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(stagingBuf, fontAtlasImage, kAtlasW, kAtlasH);
    transitionImageLayout(fontAtlasImage, VK_FORMAT_R8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkDestroyBuffer(device, stagingBuf, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image            = fontAtlasImage;
    viewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format           = VK_FORMAT_R8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCreateImageView(device, &viewInfo, nullptr, &fontAtlasView);

    VkSamplerCreateInfo sampInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampInfo.magFilter  = VK_FILTER_NEAREST;
    sampInfo.minFilter  = VK_FILTER_NEAREST;
    sampInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(device, &sampInfo, nullptr, &fontAtlasSampler);
}

void Renderer::createUIPipeline() {
    // Descriptor set layout: binding 0 = combined image sampler (font atlas)
    if (uiDescSetLayout == VK_NULL_HANDLE) {
        VkDescriptorSetLayoutBinding samplerBinding{};
        samplerBinding.binding         = 0;
        samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerBinding.descriptorCount = 1;
        samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo dslInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslInfo.bindingCount = 1;
        dslInfo.pBindings    = &samplerBinding;
        vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &uiDescSetLayout);

        // Descriptor pool: 1 set
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets       = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes    = &poolSize;
        vkCreateDescriptorPool(device, &poolInfo, nullptr, &uiDescPool);

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool     = uiDescPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &uiDescSetLayout;
        vkAllocateDescriptorSets(device, &allocInfo, &uiDescSet);

        // Write font atlas into the set
        VkDescriptorImageInfo imgInfo{};
        imgInfo.sampler     = fontAtlasSampler;
        imgInfo.imageView   = fontAtlasView;
        imgInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet           = uiDescSet;
        write.dstBinding       = 0;
        write.descriptorCount  = 1;
        write.descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo       = &imgInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        // Host-visible vertex buffer (persisently mapped)
        createBuffer(kUIMaxVertices * sizeof(UIVertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uiVertexBuffer, uiVertexBufferMemory);
        vkMapMemory(device, uiVertexBufferMemory, 0, kUIMaxVertices * sizeof(UIVertex), 0, &uiVertexMapped);
    }

    // Push constant: int mode (0=solid, 1=font)
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(int);

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &uiDescSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    vkCreatePipelineLayout(device, &layoutInfo, nullptr, &uiPipelineLayout);

    // Shaders
    auto vertCode = readFile("shaders/ui.vert.spv");
    auto fragCode = readFile("shaders/ui.frag.spv");
    VkShaderModule vertMod = createShaderModule(vertCode);
    VkShaderModule fragMod = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertMod;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragMod;
    stages[1].pName  = "main";

    // Vertex input
    auto binding = UIVertex::getBindingDescription();
    auto attrs   = UIVertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertInput.vertexBindingDescriptionCount   = 1;
    vertInput.pVertexBindingDescriptions      = &binding;
    vertInput.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    vertInput.pVertexAttributeDescriptions    = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rast{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rast.polygonMode = VK_POLYGON_MODE_FILL;
    rast.cullMode    = VK_CULL_MODE_NONE;
    rast.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rast.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // No depth test
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    // Alpha blending
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable         = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp        = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp        = VK_BLEND_OP_ADD;
    blend.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo blendState{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blendState.attachmentCount = 1;
    blendState.pAttachments    = &blend;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dynStates;

    VkGraphicsPipelineCreateInfo pipeInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeInfo.stageCount          = 2;
    pipeInfo.pStages             = stages;
    pipeInfo.pVertexInputState   = &vertInput;
    pipeInfo.pInputAssemblyState = &ia;
    pipeInfo.pViewportState      = &vp;
    pipeInfo.pRasterizationState = &rast;
    pipeInfo.pMultisampleState   = &ms;
    pipeInfo.pDepthStencilState  = &ds;
    pipeInfo.pColorBlendState    = &blendState;
    pipeInfo.pDynamicState       = &dynState;
    pipeInfo.layout              = uiPipelineLayout;
    pipeInfo.renderPass          = postRenderPass; // compatible with imguiRenderPass too
    pipeInfo.subpass             = 0;

    vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &uiPipeline);
    vkDestroyShaderModule(device, vertMod, nullptr);
    vkDestroyShaderModule(device, fragMod, nullptr);
}
