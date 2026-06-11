#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VK_CHECK(x) \
    do { \
        VkResult err = x; \
        if (err) { \
            printf("Vulkan Error: %d at %s:%d\n", err, __FILE__, __LINE__); \
            abort(); \
        } \
    } while (0)

static VkInstance g_instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_physicalDevice = VK_NULL_HANDLE;
static VkDevice g_device = VK_NULL_HANDLE;
static VkQueue g_computeQueue = VK_NULL_HANDLE;
static uint32_t g_computeFamily = 0xFFFFFFFF;
static VkCommandPool g_commandPool = VK_NULL_HANDLE;

static VkBuffer g_bufferNet = VK_NULL_HANDLE;
static VkDeviceMemory g_memoryNet = VK_NULL_HANDLE;
static VkBuffer g_stagingBufferNet = VK_NULL_HANDLE;
static VkDeviceMemory g_stagingMemoryNet = VK_NULL_HANDLE;
static size_t g_netBufferSize = 0;

static VkBuffer g_bufferPairs = VK_NULL_HANDLE;
static VkDeviceMemory g_memoryPairs = VK_NULL_HANDLE;
static VkBuffer g_stagingBufferPairs = VK_NULL_HANDLE;
static VkDeviceMemory g_stagingMemoryPairs = VK_NULL_HANDLE;
static size_t g_pairsBufferSize = 0;

uint32_t findMemoryType(VkPhysicalDevice device, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(device, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    printf("Failed to find suitable memory type!\n");
    abort();
}

void createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer* buffer, VkDeviceMemory* bufferMemory) {
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bufferInfo, NULL, buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, *buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, bufferMemory));
    vkBindBufferMemory(device, *buffer, *bufferMemory, 0);
}

void copyBuffer(VkDevice device, VkCommandPool commandPool, VkQueue computeQueue, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(computeQueue));

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void init_vulkan_context() {
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Lin Compiler GPU Dispatch";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    VK_CHECK(vkCreateInstance(&createInfo, NULL, &g_instance));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(g_instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        printf("Failed to find GPUs with Vulkan support!\n");
        abort();
    }
    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(deviceCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_instance, &deviceCount, devices);
    g_physicalDevice = devices[0];
    free(devices);

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_physicalDevice, &queueFamilyCount, NULL);
    VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(g_physicalDevice, &queueFamilyCount, queueFamilies);

    g_computeFamily = 0xFFFFFFFF;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g_computeFamily = i;
            break;
        }
    }
    free(queueFamilies);
    if (g_computeFamily == 0xFFFFFFFF) {
        printf("Failed to find compute queue family!\n");
        abort();
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = g_computeFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures = {};
    vkGetPhysicalDeviceFeatures(g_physicalDevice, &deviceFeatures);

    VkPhysicalDeviceFeatures enabledFeatures = {};
    if (deviceFeatures.shaderInt64) {
        enabledFeatures.shaderInt64 = VK_TRUE;
        printf("[GPU DISPATCH] Enabling shaderInt64 feature.\n");
    } else {
        printf("[GPU DISPATCH] Warning: shaderInt64 feature is not supported on this device!\n");
    }

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &enabledFeatures;

    VK_CHECK(vkCreateDevice(g_physicalDevice, &deviceCreateInfo, NULL, &g_device));

    vkGetDeviceQueue(g_device, g_computeFamily, 0, &g_computeQueue);

    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = g_computeFamily;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(g_device, &cmdPoolInfo, NULL, &g_commandPool));
}

void pic_gpu_cleanup() {
    if (g_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(g_device);
        
        if (g_bufferNet != VK_NULL_HANDLE) vkDestroyBuffer(g_device, g_bufferNet, NULL);
        if (g_memoryNet != VK_NULL_HANDLE) vkFreeMemory(g_device, g_memoryNet, NULL);
        if (g_stagingBufferNet != VK_NULL_HANDLE) vkDestroyBuffer(g_device, g_stagingBufferNet, NULL);
        if (g_stagingMemoryNet != VK_NULL_HANDLE) vkFreeMemory(g_device, g_stagingMemoryNet, NULL);
        
        if (g_bufferPairs != VK_NULL_HANDLE) vkDestroyBuffer(g_device, g_bufferPairs, NULL);
        if (g_memoryPairs != VK_NULL_HANDLE) vkFreeMemory(g_device, g_memoryPairs, NULL);
        if (g_stagingBufferPairs != VK_NULL_HANDLE) vkDestroyBuffer(g_device, g_stagingBufferPairs, NULL);
        if (g_stagingMemoryPairs != VK_NULL_HANDLE) vkFreeMemory(g_device, g_stagingMemoryPairs, NULL);
        
        if (g_commandPool != VK_NULL_HANDLE) vkDestroyCommandPool(g_device, g_commandPool, NULL);
        vkDestroyDevice(g_device, NULL);
    }
    if (g_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(g_instance, NULL);
    }
    
    g_instance = VK_NULL_HANDLE;
    g_physicalDevice = VK_NULL_HANDLE;
    g_device = VK_NULL_HANDLE;
    g_computeQueue = VK_NULL_HANDLE;
    g_computeFamily = 0xFFFFFFFF;
    g_commandPool = VK_NULL_HANDLE;
    
    g_bufferNet = VK_NULL_HANDLE;
    g_memoryNet = VK_NULL_HANDLE;
    g_stagingBufferNet = VK_NULL_HANDLE;
    g_stagingMemoryNet = VK_NULL_HANDLE;
    g_netBufferSize = 0;
    
    g_bufferPairs = VK_NULL_HANDLE;
    g_memoryPairs = VK_NULL_HANDLE;
    g_stagingBufferPairs = VK_NULL_HANDLE;
    g_stagingMemoryPairs = VK_NULL_HANDLE;
    g_pairsBufferSize = 0;
    
    printf("[GPU DISPATCH] Cleaned up Vulkan resources successfully.\n");
}

void pic_gpu_dispatch(int32_t* netPtr, int32_t* activePairsPtr, int32_t numPairs, int32_t* alPtr, const char* spirvPath) {
    printf("[GPU DISPATCH] Launching parallel kernel for %d active pairs (SPIR-V: %s)\n", numPairs, spirvPath);

    if (numPairs == 0) {
        printf("[GPU DISPATCH] No active pairs to process.\n");
        return;
    }

    FILE *f = fopen(spirvPath, "rb");
    if (!f) {
        printf("[GPU DISPATCH] Error: SPIR-V module %s not found\n", spirvPath);
        return;
    }
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    rewind(f);
    uint32_t* shaderCode = (uint32_t*)malloc(fileSize);
    if (fread(shaderCode, 1, fileSize, f) != fileSize) {
        printf("[GPU DISPATCH] Error reading SPIR-V module\n");
        fclose(f);
        free(shaderCode);
        return;
    }
    fclose(f);

    // Initialize Vulkan Context if not already cached
    if (g_instance == VK_NULL_HANDLE) {
        init_vulkan_context();
    }

    VkDevice device = g_device;
    VkPhysicalDevice physicalDevice = g_physicalDevice;
    VkQueue computeQueue = g_computeQueue;
    VkCommandPool commandPool = g_commandPool;

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = fileSize;
    shaderModuleCreateInfo.pCode = shaderCode;

    VkShaderModule computeShaderModule;
    VK_CHECK(vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &computeShaderModule));

    // Staging and Device Net Buffers caching/initialization
    size_t netBufferSize = (1000000 * 4 + 1000000 * 2 + 10) * 4;
    if (g_bufferNet == VK_NULL_HANDLE) {
        g_netBufferSize = netBufferSize;
        createBuffer(device, physicalDevice, g_netBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &g_stagingBufferNet, &g_stagingMemoryNet);
        createBuffer(device, physicalDevice, g_netBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &g_bufferNet, &g_memoryNet);
        printf("[GPU DISPATCH] Initialized persistent net buffers (%zu bytes)\n", g_netBufferSize);
    }

    // Staging and Device Active Pairs Buffers caching/resizing
    size_t requiredPairsBufferSize = (numPairs > 0 ? numPairs : 1) * 2 * sizeof(int32_t);
    if (g_bufferPairs == VK_NULL_HANDLE || g_pairsBufferSize < requiredPairsBufferSize) {
        if (g_bufferPairs != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, g_bufferPairs, NULL);
            vkFreeMemory(device, g_memoryPairs, NULL);
            vkDestroyBuffer(device, g_stagingBufferPairs, NULL);
            vkFreeMemory(device, g_stagingMemoryPairs, NULL);
        }
        
        size_t newCapacity = g_pairsBufferSize;
        if (newCapacity == 0) newCapacity = 1024 * 2 * sizeof(int32_t);
        while (newCapacity < requiredPairsBufferSize) {
            newCapacity *= 2;
        }
        g_pairsBufferSize = newCapacity;
        
        createBuffer(device, physicalDevice, g_pairsBufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &g_stagingBufferPairs, &g_stagingMemoryPairs);
        createBuffer(device, physicalDevice, g_pairsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &g_bufferPairs, &g_memoryPairs);
        printf("[GPU DISPATCH] Allocated/Resized persistent pairs buffers (%zu bytes)\n", g_pairsBufferSize);
    }

    VkBuffer stagingBufferNet = g_stagingBufferNet;
    VkDeviceMemory stagingMemoryNet = g_stagingMemoryNet;
    VkBuffer bufferNet = g_bufferNet;
    VkDeviceMemory memoryNet = g_memoryNet;

    VkBuffer stagingBufferPairs = g_stagingBufferPairs;
    VkDeviceMemory stagingMemoryPairs = g_stagingMemoryPairs;
    VkBuffer bufferPairs = g_bufferPairs;
    VkDeviceMemory memoryPairs = g_memoryPairs;

    // We still allocate staging and device buffer for numPairs since it's just a single int32_t, but now 2 * sizeof(int32_t) to include allocation pointer
    VkBuffer stagingBufferNumPairs, bufferNumPairs;
    VkDeviceMemory stagingMemoryNumPairs, memoryNumPairs;
    createBuffer(device, physicalDevice, 2 * sizeof(int32_t), VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stagingBufferNumPairs, &stagingMemoryNumPairs);
    createBuffer(device, physicalDevice, 2 * sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &bufferNumPairs, &memoryNumPairs);

    void* data;

    // Copy the active pairs from host
    vkMapMemory(device, stagingMemoryPairs, 0, requiredPairsBufferSize, 0, &data);
    if (numPairs > 0 && activePairsPtr != NULL) {
        memcpy(data, activePairsPtr, requiredPairsBufferSize);
    }
    vkUnmapMemory(device, stagingMemoryPairs);

    // Copy the network graph from host
    vkMapMemory(device, stagingMemoryNet, 0, netBufferSize, 0, &data);
    if (netPtr != NULL) {
        memcpy(data, netPtr, netBufferSize);
    }
    vkUnmapMemory(device, stagingMemoryNet);

    struct {
        int32_t numPairs;
        int32_t al;
    } gpuState;
    gpuState.numPairs = numPairs;
    gpuState.al = alPtr ? *alPtr : 0;

    vkMapMemory(device, stagingMemoryNumPairs, 0, 2 * sizeof(int32_t), 0, &data);
    memcpy(data, &gpuState, 2 * sizeof(int32_t));
    vkUnmapMemory(device, stagingMemoryNumPairs);

    copyBuffer(device, commandPool, computeQueue, stagingBufferNet, bufferNet, netBufferSize);
    copyBuffer(device, commandPool, computeQueue, stagingBufferPairs, bufferPairs, requiredPairsBufferSize);
    copyBuffer(device, commandPool, computeQueue, stagingBufferNumPairs, bufferNumPairs, 2 * sizeof(int32_t));

    VkDescriptorSetLayoutBinding bindings[3];
    for (int i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = NULL;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout));

    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool));

    VkDescriptorSetAllocateInfo allocInfoDesc = {};
    allocInfoDesc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfoDesc.descriptorPool = descriptorPool;
    allocInfoDesc.descriptorSetCount = 1;
    allocInfoDesc.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfoDesc, &descriptorSet));

    VkDescriptorBufferInfo bufferInfos[3];
    bufferInfos[0].buffer = bufferNet; bufferInfos[0].offset = 0; bufferInfos[0].range = netBufferSize;
    bufferInfos[1].buffer = bufferPairs; bufferInfos[1].offset = 0; bufferInfos[1].range = requiredPairsBufferSize;
    bufferInfos[2].buffer = bufferNumPairs; bufferInfos[2].offset = 0; bufferInfos[2].range = 2 * sizeof(int32_t);

    VkWriteDescriptorSet descriptorWrites[3];
    for (int i = 0; i < 3; i++) {
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].pNext = NULL;
        descriptorWrites[i].dstSet = descriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfos[i];
        descriptorWrites[i].pImageInfo = NULL;
        descriptorWrites[i].pTexelBufferView = NULL;
    }
    vkUpdateDescriptorSets(device, 3, descriptorWrites, 0, NULL);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;

    VkPipelineLayout pipelineLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &pipelineLayoutInfo, NULL, &pipelineLayout));

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = computeShaderModule;
    pipelineInfo.stage.pName = "pic_kernel"; 
    pipelineInfo.layout = pipelineLayout;

    VkPipeline computePipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &computePipeline));

    VkCommandBufferAllocateInfo cmdAllocInfo = {};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer));

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, NULL);

    uint32_t groupCountX = (numPairs + 255) / 256;
    vkCmdDispatch(commandBuffer, groupCountX, 1, 1);
    
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(computeQueue));

    // Read results back to host
    copyBuffer(device, commandPool, computeQueue, bufferNet, stagingBufferNet, netBufferSize);
    copyBuffer(device, commandPool, computeQueue, bufferNumPairs, stagingBufferNumPairs, 2 * sizeof(int32_t));

    vkMapMemory(device, stagingMemoryNet, 0, netBufferSize, 0, &data);
    if (netPtr != NULL) {
        memcpy(netPtr, data, netBufferSize);
    }
    vkUnmapMemory(device, stagingMemoryNet);

    vkMapMemory(device, stagingMemoryNumPairs, 0, 2 * sizeof(int32_t), 0, &data);
    memcpy(&gpuState, data, 2 * sizeof(int32_t));
    vkUnmapMemory(device, stagingMemoryNumPairs);

    if (alPtr) {
        *alPtr = gpuState.al;
    }

    printf("[GPU DISPATCH] Completed successfully.\n");

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    vkDestroyPipeline(device, computePipeline, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
    
    vkDestroyBuffer(device, bufferNumPairs, NULL); vkFreeMemory(device, memoryNumPairs, NULL);
    vkDestroyBuffer(device, stagingBufferNumPairs, NULL); vkFreeMemory(device, stagingMemoryNumPairs, NULL);

    vkDestroyShaderModule(device, computeShaderModule, NULL);
    free(shaderCode);
}
