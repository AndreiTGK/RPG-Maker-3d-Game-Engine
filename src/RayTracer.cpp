#include "RayTracer.hpp"
#include <glm/glm.hpp>
#include "EngineLog.hpp"
#include "Renderer.hpp"

void RayTracer::init(VkDevice logicalDevice, VkPhysicalDevice physDevice) {
    device = logicalDevice;
    physicalDevice = physDevice;
    loadFunctionPointers();
    rtDescriptorSet = VK_NULL_HANDLE;
    initialized = true;
}

bool RayTracer::isReady() {
    return initialized
        && vkGetBufferDeviceAddressKHR             != nullptr
        && vkCreateAccelerationStructureKHR        != nullptr
        && vkDestroyAccelerationStructureKHR       != nullptr
        && vkGetAccelerationStructureBuildSizesKHR != nullptr
        && vkGetAccelerationStructureDeviceAddressKHR != nullptr
        && vkCmdBuildAccelerationStructuresKHR     != nullptr
        && vkCreateRayTracingPipelinesKHR          != nullptr
        && vkGetRayTracingShaderGroupHandlesKHR    != nullptr
        && vkCmdTraceRaysKHR                       != nullptr;
}

void RayTracer::loadFunctionPointers() {
    vkGetBufferDeviceAddressKHR = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR"));
    vkCreateAccelerationStructureKHR = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR"));
    vkDestroyAccelerationStructureKHR = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR"));
    vkGetAccelerationStructureBuildSizesKHR = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR"));
    vkGetAccelerationStructureDeviceAddressKHR = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR"));
    vkCmdBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR"));
    vkBuildAccelerationStructuresKHR = reinterpret_cast<PFN_vkBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device, "vkBuildAccelerationStructuresKHR"));
    vkCreateRayTracingPipelinesKHR = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR"));
    vkGetRayTracingShaderGroupHandlesKHR = reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR"));
    vkCmdTraceRaysKHR = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR"));
}
VkDeviceAddress RayTracer::getBufferDeviceAddress(VkBuffer buffer) {
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddressKHR(device, &info);
}

uint32_t RayTracer::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    THROW_ENGINE_ERROR("Eroare memorie: Nu s-a gasit tipul de memorie potrivit pentru RT.");
}

void RayTracer::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare la crearea buffer-ului in RayTracer!");
    }

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, buffer, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &allocFlags;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare la alocarea memoriei in RayTracer!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void RayTracer::buildBLAS(const std::string& name, VkBuffer vertexBuffer, uint32_t vertexCount, VkBuffer indexBuffer, uint32_t indexCount, VkCommandPool commandPool, VkQueue graphicsQueue) {
    // Destroy existing BLAS for this name (e.g. mesh reload)
    if (blasMap.count(name)) {
        auto& old = blasMap[name];
        if (old.as != VK_NULL_HANDLE) vkDestroyAccelerationStructureKHR(device, old.as, nullptr);
        if (old.buf != VK_NULL_HANDLE) { vkDestroyBuffer(device, old.buf, nullptr); vkFreeMemory(device, old.mem, nullptr); }
    }

    BlasEntry& entry = blasMap[name];
    VkDeviceAddress vertexAddress = getBufferDeviceAddress(vertexBuffer);
    VkDeviceAddress indexAddress = getBufferDeviceAddress(indexBuffer);

    VkAccelerationStructureGeometryTrianglesDataKHR triangles{};
    triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride = sizeof(Vertex);
    triangles.maxVertex = vertexCount - 1;
    triangles.indexType = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress = indexAddress;

    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triangles;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    uint32_t numTriangles = indexCount / 3;
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &numTriangles, &sizeInfo);

    createBuffer(sizeInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, entry.buf, entry.mem);

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = entry.buf;
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &entry.as);

    VkBuffer scratchBuffer;
    VkDeviceMemory scratchBufferMemory;
    createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, scratchBuffer, scratchBufferMemory);

    buildInfo.dstAccelerationStructure = entry.as;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(scratchBuffer);

    VkAccelerationStructureBuildRangeInfoKHR buildRangeInfo{};
    buildRangeInfo.primitiveCount = numTriangles;
    buildRangeInfo.primitiveOffset = 0;
    buildRangeInfo.firstVertex = 0;
    buildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> pBuildRangeInfos = {&buildRangeInfo};

    VkCommandBufferAllocateInfo allocCmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocCmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCmdInfo.commandPool = commandPool;
    allocCmdInfo.commandBufferCount = 1;
    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(device, &allocCmdInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, pBuildRangeInfos.data());

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuf;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
    vkDestroyBuffer(device, scratchBuffer, nullptr);
    vkFreeMemory(device, scratchBufferMemory, nullptr);
}


VkTransformMatrixKHR RayTracer::glmMat4ToVkTransformMatrix(const glm::mat4& matrix) {
    // Matricea Vulkan KHR asteapta un format Row-Major 3x4
    // glm::mat4 este Column-Major 4x4, de aceea extragem valorile transpus
    VkTransformMatrixKHR transformMatrix;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            transformMatrix.matrix[row][col] = matrix[col][row];
        }
    }
    return transformMatrix;
}

void RayTracer::buildTLAS(const std::vector<RtInstance>& rtInstances, const std::vector<std::string>& meshNames, VkCommandPool commandPool, VkQueue graphicsQueue) {
    if (rtInstances.empty() || blasMap.empty()) return;

    // Build one VkAccelerationStructureInstanceKHR per RtInstance, skipping any
    // whose mesh has no BLAS yet (e.g. entity spawned before mesh loaded).
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(rtInstances.size());

    for (const auto& ri : rtInstances) {
        auto blasIt = blasMap.find(ri.meshName);
        if (blasIt == blasMap.end() || blasIt->second.as == VK_NULL_HANDLE) continue;

        // Mesh index = position in meshNames vector (caller must pass the same order used for VB/IB descriptor arrays)
        uint32_t meshIdx = 0;
        auto nameIt = std::find(meshNames.begin(), meshNames.end(), ri.meshName);
        if (nameIt != meshNames.end()) meshIdx = static_cast<uint32_t>(nameIt - meshNames.begin());

        VkAccelerationStructureDeviceAddressInfoKHR addrInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        addrInfo.accelerationStructure = blasIt->second.as;
        VkDeviceAddress blasAddr = vkGetAccelerationStructureDeviceAddressKHR(device, &addrInfo);

        // Pack: bits[23:16] = mesh index (0-255), bits[15:0] = texture index
        uint32_t customIndex = ((meshIdx & 0xFF) << 16) | (ri.textureIndex & 0xFFFF);

        VkAccelerationStructureInstanceKHR inst{};
        inst.transform = glmMat4ToVkTransformMatrix(ri.transform);
        inst.instanceCustomIndex = customIndex;
        inst.mask = 0xFF;
        inst.instanceShaderBindingTableRecordOffset = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        inst.accelerationStructureReference = blasAddr;
        instances.push_back(inst);
    }

    if (instances.empty()) return;
    uint32_t numInstances = static_cast<uint32_t>(instances.size());

    // --- Geometry descriptor (device address filled in below) ---
    VkAccelerationStructureGeometryInstancesDataKHR instancesData{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    instancesData.arrayOfPointers = VK_FALSE;

    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances = instancesData;

    // Build info template (mode set below)
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    // ALLOW_UPDATE lets subsequent frames use cheap UPDATE mode instead of full BUILD
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                      VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;

    // --- Decide: realloc, full BUILD, or cheap UPDATE ---
    //
    // Vulkan spec: UPDATE mode requires primitiveCount to be IDENTICAL to the original build.
    // So UPDATE is only valid when entity count has not changed. Any count change (even a
    // decrease) must use full BUILD mode. We separate reallocation (rare) from rebuild.
    //
    // needRealloc: TLAS buffers don't exist or are too small → destroy + recreate + full build
    // canUpdate  : entity count unchanged → UPDATE mode (no GPU structure recreation, very fast)
    // else       : count changed but within capacity → full BUILD into existing TLAS handle

    bool needRealloc = (tlas == VK_NULL_HANDLE) || (numInstances > tlasAllocatedInstances);
    bool canUpdate   = !needRealloc && (numInstances == tlasCurrentInstances);

    if (needRealloc) {
        // Free old AS and buffers
        if (tlas != VK_NULL_HANDLE) {
            vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
            vkDestroyBuffer(device, tlasBuffer, nullptr);
            vkFreeMemory(device, tlasBufferMemory, nullptr);
            tlas = VK_NULL_HANDLE;
        }
        if (tlasInstancesBuffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, tlasInstancesBuffer, nullptr);
            vkFreeMemory(device, tlasInstancesBufferMemory, nullptr);
            tlasInstancesBuffer = VK_NULL_HANDLE;
        }
        if (tlasPersistentScratch != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, tlasPersistentScratch, nullptr);
            vkFreeMemory(device, tlasPersistentScratchMemory, nullptr);
            tlasPersistentScratch = VK_NULL_HANDLE;
        }

        // Over-provision so small entity additions don't immediately trigger another realloc
        uint32_t allocCount = std::max(numInstances * 2u, 64u);

        // Query sizes for the provisioned count
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                                &buildInfo, &allocCount, &sizeInfo);

        // TLAS storage buffer + structure
        createBuffer(sizeInfo.accelerationStructureSize,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tlasBuffer, tlasBufferMemory);

        VkAccelerationStructureCreateInfoKHR createInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        createInfo.buffer = tlasBuffer;
        createInfo.size   = sizeInfo.accelerationStructureSize;
        createInfo.type   = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        vkCreateAccelerationStructureKHR(device, &createInfo, nullptr, &tlas);

        // Persistent host-visible instance buffer
        createBuffer(allocCount * sizeof(VkAccelerationStructureInstanceKHR),
                     VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     tlasInstancesBuffer, tlasInstancesBufferMemory);

        // Persistent device-local scratch (large enough for both build and update)
        VkDeviceSize scratchSize = std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize);
        createBuffer(scratchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     tlasPersistentScratch, tlasPersistentScratchMemory);

        tlasAllocatedInstances = allocCount;
        rtDescriptorsDirty = true;  // New TLAS handle — descriptor must be re-written
    }

    // --- Upload instance transforms to persistent host-visible buffer ---
    void* data;
    vkMapMemory(device, tlasInstancesBufferMemory, 0,
                numInstances * sizeof(VkAccelerationStructureInstanceKHR), 0, &data);
    memcpy(data, instances.data(), numInstances * sizeof(VkAccelerationStructureInstanceKHR));
    vkUnmapMemory(device, tlasInstancesBufferMemory);

    // Wire up device address now that the buffer exists
    instancesData.data.deviceAddress = getBufferDeviceAddress(tlasInstancesBuffer);
    geometry.geometry.instances = instancesData;
    buildInfo.pGeometries = &geometry;

    // Select build mode
    if (canUpdate) {
        // Fast path: same entity count, only transforms changed
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.srcAccelerationStructure = tlas;
    } else {
        // Full build: entity count changed (or first build)
        // Reuses existing TLAS handle+buffer when within allocated capacity (no realloc)
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    }
    buildInfo.dstAccelerationStructure = tlas;
    buildInfo.scratchData.deviceAddress = getBufferDeviceAddress(tlasPersistentScratch);

    // --- Submit build command ---
    VkCommandBufferAllocateInfo allocCmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocCmdInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCmdInfo.commandPool        = commandPool;
    allocCmdInfo.commandBufferCount = 1;
    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(device, &allocCmdInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = numInstances;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> pRangeInfos = {&rangeInfo};
    vkCmdBuildAccelerationStructuresKHR(cmdBuf, 1, &buildInfo, pRangeInfos.data());

    vkEndCommandBuffer(cmdBuf);

    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmdBuf;
    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
    tlasCurrentInstances = numInstances;
}

void RayTracer::cleanup() {
    if (!initialized) return;
    destroyStorageImage();

    if (tlas != VK_NULL_HANDLE) {
        vkDestroyAccelerationStructureKHR(device, tlas, nullptr);
        vkDestroyBuffer(device, tlasBuffer, nullptr);
        vkFreeMemory(device, tlasBufferMemory, nullptr);
        vkDestroyBuffer(device, tlasInstancesBuffer, nullptr);
        vkFreeMemory(device, tlasInstancesBufferMemory, nullptr);
    }

    if (tlasPersistentScratch != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, tlasPersistentScratch, nullptr);
        vkFreeMemory(device, tlasPersistentScratchMemory, nullptr);
    }

    for (auto& [name, entry] : blasMap) {
        if (entry.as  != VK_NULL_HANDLE) vkDestroyAccelerationStructureKHR(device, entry.as, nullptr);
        if (entry.buf != VK_NULL_HANDLE) { vkDestroyBuffer(device, entry.buf, nullptr); vkFreeMemory(device, entry.mem, nullptr); }
    }
    blasMap.clear();

    if (sbtBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, sbtBuffer, nullptr);
        vkFreeMemory(device, sbtBufferMemory, nullptr);
    }
    vkDestroyPipeline(device, rtPipeline, nullptr);
    vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);
    vkDestroyDescriptorPool(device, rtDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, rtDescriptorSetLayout, nullptr);
}

void RayTracer::createStorageImage(uint32_t width, uint32_t height) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkCreateImage(device, &imageInfo, nullptr, &storageImage);

    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(device, storageImage, &memReqs);

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkAllocateMemory(device, &allocInfo, nullptr, &storageImageMemory);
    vkBindImageMemory(device, storageImage, storageImageMemory, 0);

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    viewInfo.image = storageImage;

    vkCreateImageView(device, &viewInfo, nullptr, &storageImageView);
}

void RayTracer::destroyStorageImage() {
    if (storageImage != VK_NULL_HANDLE) {
        vkDestroyImageView(device, storageImageView, nullptr);
        vkDestroyImage(device, storageImage, nullptr);
        vkFreeMemory(device, storageImageMemory, nullptr);

        storageImage = VK_NULL_HANDLE;
        storageImageView = VK_NULL_HANDLE;
        storageImageMemory = VK_NULL_HANDLE;
    }
}

void RayTracer::createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding asLayoutBinding{};
    asLayoutBinding.binding = 0;
    asLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    asLayoutBinding.descriptorCount = 1;
    asLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    VkDescriptorSetLayoutBinding storageImageLayoutBinding{};
    storageImageLayoutBinding.binding = 1;
    storageImageLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    storageImageLayoutBinding.descriptorCount = 1;
    storageImageLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    VkDescriptorSetLayoutBinding uniformBufferBinding{};
    uniformBufferBinding.binding = 2;
    uniformBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBufferBinding.descriptorCount = 1;
    uniformBufferBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

    VkDescriptorSetLayoutBinding vertexBufferBinding{};
    vertexBufferBinding.binding = 3;
    vertexBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vertexBufferBinding.descriptorCount = MAX_RT_MESHES;
    vertexBufferBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    VkDescriptorSetLayoutBinding indexBufferBinding{};
    indexBufferBinding.binding = 4;
    indexBufferBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    indexBufferBinding.descriptorCount = MAX_RT_MESHES;
    indexBufferBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    VkDescriptorSetLayoutBinding textureBinding{};
    textureBinding.binding = 5;
    textureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureBinding.descriptorCount = 100;
    textureBinding.pImmutableSamplers = nullptr;
    textureBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        asLayoutBinding, storageImageLayoutBinding, uniformBufferBinding, vertexBufferBinding, indexBufferBinding, textureBinding
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    VkDescriptorBindingFlags bindFlags[6] = {0, 0, 0,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};
    flagsInfo.bindingCount = 6;
    flagsInfo.pBindingFlags = bindFlags;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.pNext = &flagsInfo;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &rtDescriptorSetLayout);
}

void RayTracer::createDescriptorPool() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_RT_MESHES * 2},
        // MODIFICARE IMPORTANTĂ: Crestem si aici capacitatea pool-ului la 100 de imagini
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 100}
    };

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;

    vkCreateDescriptorPool(device, &poolInfo, nullptr, &rtDescriptorPool);
}

void RayTracer::createDescriptorSets(VkBuffer uniformBuffer, VkDeviceSize uboSize, const std::vector<std::pair<VkBuffer,VkBuffer>>& meshBuffers, const std::vector<VkImageView>& textureImageViews, const std::vector<VkSampler>& textureSamplers) {
    if (tlas == VK_NULL_HANDLE || uniformBuffer == VK_NULL_HANDLE || meshBuffers.empty() || textureImageViews.empty()) return;

    if (rtDescriptorSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool = rtDescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &rtDescriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &rtDescriptorSet) != VK_SUCCESS) {
            THROW_ENGINE_ERROR("Eroare la alocarea Descriptor Set!");
        }
        rtDescriptorsDirty = true;
    }

    // Skip the expensive vkUpdateDescriptorSets when nothing has changed.
    // rtDescriptorsDirty is set to true by buildTLAS after a full rebuild.
    if (!rtDescriptorsDirty) return;

    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures = &tlas;

    VkWriteDescriptorSet asWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    asWrite.dstSet = rtDescriptorSet;
    asWrite.dstBinding = 0;
    asWrite.descriptorCount = 1;
    asWrite.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    asWrite.pNext = &descASInfo;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = storageImageView;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet imageWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    imageWrite.dstSet = rtDescriptorSet;
    imageWrite.dstBinding = 1;
    imageWrite.descriptorCount = 1;
    imageWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    imageWrite.pImageInfo = &imageInfo;

    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = uniformBuffer;
    uboInfo.offset = 0;
    uboInfo.range = uboSize;

    VkWriteDescriptorSet uboWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    uboWrite.dstSet = rtDescriptorSet;
    uboWrite.dstBinding = 2;
    uboWrite.descriptorCount = 1;
    uboWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboWrite.pBufferInfo = &uboInfo;

    uint32_t meshCount = std::min(static_cast<uint32_t>(meshBuffers.size()), MAX_RT_MESHES);
    std::vector<VkDescriptorBufferInfo> vbInfos(meshCount), ibInfos(meshCount);
    for (uint32_t i = 0; i < meshCount; i++) {
        vbInfos[i] = {meshBuffers[i].first,  0, VK_WHOLE_SIZE};
        ibInfos[i] = {meshBuffers[i].second, 0, VK_WHOLE_SIZE};
    }

    VkWriteDescriptorSet vertexWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    vertexWrite.dstSet = rtDescriptorSet;
    vertexWrite.dstBinding = 3;
    vertexWrite.dstArrayElement = 0;
    vertexWrite.descriptorCount = meshCount;
    vertexWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vertexWrite.pBufferInfo = vbInfos.data();

    VkWriteDescriptorSet indexWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    indexWrite.dstSet = rtDescriptorSet;
    indexWrite.dstBinding = 4;
    indexWrite.dstArrayElement = 0;
    indexWrite.descriptorCount = meshCount;
    indexWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    indexWrite.pBufferInfo = ibInfos.data();

    std::vector<VkDescriptorImageInfo> texImageInfos(textureImageViews.size());
    for (size_t i = 0; i < textureImageViews.size(); ++i) {
        texImageInfos[i].imageView = textureImageViews[i];
        texImageInfos[i].sampler = textureSamplers[i];
        texImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkWriteDescriptorSet textureWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    textureWrite.dstSet = rtDescriptorSet;
    textureWrite.dstBinding = 5;
    textureWrite.dstArrayElement = 0;
    textureWrite.descriptorCount = static_cast<uint32_t>(texImageInfos.size()); // Legăm TOT array-ul
    textureWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    textureWrite.pImageInfo = texImageInfos.data();

    std::vector<VkWriteDescriptorSet> writes = {asWrite, imageWrite, uboWrite, vertexWrite, indexWrite, textureWrite};
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    rtDescriptorsDirty = false;
}

std::vector<char> RayTracer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) THROW_ENGINE_ERROR("Eroare la citirea shaderului: " + filename);
    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

VkShaderModule RayTracer::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule sm;
    vkCreateShaderModule(device, &ci, nullptr, &sm);
    return sm;
}

void RayTracer::createRayTracingPipeline() {
    auto raygenCode = readFile("shaders/raygen.spv");
    auto missCode = readFile("shaders/miss.spv");
    auto shadowMissCode = readFile("shaders/shadow.rmiss.spv");
    auto chitCode = readFile("shaders/closesthit.spv");

    VkShaderModule raygenModule = createShaderModule(raygenCode);
    VkShaderModule missModule = createShaderModule(missCode);
    VkShaderModule shadowMissModule = createShaderModule(shadowMissCode);
    VkShaderModule chitModule = createShaderModule(chitCode);

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    VkPipelineShaderStageCreateInfo raygenStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    raygenStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    raygenStage.module = raygenModule;
    raygenStage.pName = "main";
    shaderStages.push_back(raygenStage);

    VkPipelineShaderStageCreateInfo missStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    missStage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    missStage.module = missModule;
    missStage.pName = "main";
    shaderStages.push_back(missStage);

    VkPipelineShaderStageCreateInfo shadowMissStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    shadowMissStage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    shadowMissStage.module = shadowMissModule;
    shadowMissStage.pName = "main";
    shaderStages.push_back(shadowMissStage);

    VkPipelineShaderStageCreateInfo chitStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    chitStage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    chitStage.module = chitModule;
    chitStage.pName = "main";
    shaderStages.push_back(chitStage);

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;

    VkRayTracingShaderGroupCreateInfoKHR group0{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group0.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group0.generalShader = 0;
    group0.closestHitShader = VK_SHADER_UNUSED_KHR;
    group0.anyHitShader = VK_SHADER_UNUSED_KHR;
    group0.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(group0);

    VkRayTracingShaderGroupCreateInfoKHR group1{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group1.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group1.generalShader = 1;
    group1.closestHitShader = VK_SHADER_UNUSED_KHR;
    group1.anyHitShader = VK_SHADER_UNUSED_KHR;
    group1.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(group1);

    VkRayTracingShaderGroupCreateInfoKHR group2{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group2.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group2.generalShader = 2;
    group2.closestHitShader = VK_SHADER_UNUSED_KHR;
    group2.anyHitShader = VK_SHADER_UNUSED_KHR;
    group2.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(group2);

    VkRayTracingShaderGroupCreateInfoKHR group3{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group3.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group3.generalShader = VK_SHADER_UNUSED_KHR;
    group3.closestHitShader = 3;
    group3.anyHitShader = VK_SHADER_UNUSED_KHR;
    group3.intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroups.push_back(group3);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &rtDescriptorSetLayout;
    vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &rtPipelineLayout);

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    pipelineInfo.pGroups = shaderGroups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 2;
    pipelineInfo.layout = rtPipelineLayout;

    auto pfn_vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");
    if (!pfn_vkCreateRayTracingPipelinesKHR) {
        THROW_ENGINE_ERROR("Eroare hardware: Nu s-a putut incarca vkCreateRayTracingPipelinesKHR!");
    }

    if (pfn_vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &rtPipeline) != VK_SUCCESS) {
        THROW_ENGINE_ERROR("Eroare interna: Placa video a refuzat crearea conductei RTX!");
    }

    vkDestroyShaderModule(device, raygenModule, nullptr);
    vkDestroyShaderModule(device, missModule, nullptr);
    vkDestroyShaderModule(device, shadowMissModule, nullptr);
    vkDestroyShaderModule(device, chitModule, nullptr);
}

uint32_t RayTracer::alignUp(uint32_t size, uint32_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

void RayTracer::createShaderBindingTable() {
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &rtProperties;
    vkGetPhysicalDeviceProperties2(physicalDevice, &prop2);

    uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;

    uint32_t groupCount = 4;
    uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

    raygenRegion.stride = alignUp(handleSizeAligned, baseAlignment);
    raygenRegion.size = raygenRegion.stride;

    missRegion.stride = handleSizeAligned;
    missRegion.size = alignUp(missRegion.stride * 2, baseAlignment);

    hitRegion.stride = handleSizeAligned;
    hitRegion.size = alignUp(hitRegion.stride, baseAlignment);

    auto pfn_vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");

    uint32_t dataSize = groupCount * handleSize;
    std::vector<uint8_t> handles(dataSize);
    pfn_vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline, 0, groupCount, dataSize, handles.data());

    uint32_t sbtSize = raygenRegion.size + missRegion.size + hitRegion.size;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = sbtSize;
    bufferInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bufferInfo, nullptr, &sbtBuffer);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, sbtBuffer, &memReqs);

    VkMemoryAllocateFlagsInfo allocFlags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    allocFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.pNext = &allocFlags;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vkAllocateMemory(device, &allocInfo, nullptr, &sbtBufferMemory);
    vkBindBufferMemory(device, sbtBuffer, sbtBufferMemory, 0);

    VkDeviceAddress sbtAddress = getBufferDeviceAddress(sbtBuffer);

    raygenRegion.deviceAddress = sbtAddress;
    missRegion.deviceAddress = sbtAddress + raygenRegion.size;
    hitRegion.deviceAddress = sbtAddress + raygenRegion.size + missRegion.size;

    void* mappedData;
    vkMapMemory(device, sbtBufferMemory, 0, sbtSize, 0, &mappedData);
    uint8_t* pData = reinterpret_cast<uint8_t*>(mappedData);

    memcpy(pData, handles.data(), handleSize);

    uint8_t* pMissData = pData + raygenRegion.size;
    memcpy(pMissData, handles.data() + handleSize, handleSize);
    memcpy(pMissData + missRegion.stride, handles.data() + 2 * handleSize, handleSize);

    uint8_t* pHitData = pData + raygenRegion.size + missRegion.size;
    memcpy(pHitData, handles.data() + 3 * handleSize, handleSize);

    vkUnmapMemory(device, sbtBufferMemory);
}
