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

void pic_gpu_dispatch(int32_t* netPtr, int32_t* activePairsPtr, int32_t numPairs) {
    printf("[GPU DISPATCH] Launching parallel kernel for %d active pairs\n", numPairs);

    if (numPairs == 0) {
        printf("[GPU DISPATCH] No active pairs to process.\n");
        return;
    }

    FILE *f = fopen("linc_out.spv", "rb");
    if (!f) {
        printf("[GPU DISPATCH] Error: SPIR-V module linc_out.spv not found\n");
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
    
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&createInfo, NULL, &instance));

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        printf("Failed to find GPUs with Vulkan support!\n");
        abort();
    }
    VkPhysicalDevice* devices = (VkPhysicalDevice*)malloc(deviceCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    VkPhysicalDevice physicalDevice = devices[0];

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, NULL);
    VkQueueFamilyProperties* queueFamilies = (VkQueueFamilyProperties*)malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);

    int computeFamily = -1;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeFamily = i;
            break;
        }
    }
    if (computeFamily == -1) {
        printf("Failed to find compute queue family!\n");
        abort();
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo deviceCreateInfo = {};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;

    VkDevice device;
    VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, NULL, &device));

    VkQueue computeQueue;
    vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);

    VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = fileSize;
    shaderModuleCreateInfo.pCode = shaderCode;

    VkShaderModule computeShaderModule;
    VK_CHECK(vkCreateShaderModule(device, &shaderModuleCreateInfo, NULL, &computeShaderModule));

    // Create 3 buffers: netPtr, activePairs, numPairs
    VkBuffer bufferNet, bufferPairs, bufferNumPairs;
    VkDeviceMemory memoryNet, memoryPairs, memoryNumPairs;
    
    // Allocate 16MB for the network graph (simulation size for MVP)
    size_t netBufferSize = 16 * 1024 * 1024;
    size_t pairsBufferSize = numPairs * sizeof(int32_t);

    createBuffer(device, physicalDevice, netBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &bufferNet, &memoryNet);
    createBuffer(device, physicalDevice, pairsBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &bufferPairs, &memoryPairs);
    createBuffer(device, physicalDevice, sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &bufferNumPairs, &memoryNumPairs);

    void* data;
    
    // Copy the dummy simulation data into the active pairs buffer
    vkMapMemory(device, memoryPairs, 0, pairsBufferSize, 0, &data);
    int32_t* pairsData = (int32_t*)data;
    for (int i = 0; i < numPairs; i++) {
        // Each pair occupies 4 elements in the net buffer
        pairsData[i] = i * 4;
    }
    vkUnmapMemory(device, memoryPairs);

    // Initialize the network with dummy math ops (add, 10, i, out)
    vkMapMemory(device, memoryNet, 0, netBufferSize, 0, &data);
    int32_t* netData = (int32_t*)data;
    for (int i = 0; i < numPairs; i++) {
        int idx = i * 4;
        netData[idx] = 0; // opcode 0
        netData[idx + 1] = 10; // lhs
        netData[idx + 2] = i;  // rhs
        netData[idx + 3] = -1; // output initialized to -1
    }
    vkUnmapMemory(device, memoryNet);

    vkMapMemory(device, memoryNumPairs, 0, sizeof(int32_t), 0, &data);
    memcpy(data, &numPairs, sizeof(int32_t));
    vkUnmapMemory(device, memoryNumPairs);

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
    bufferInfos[1].buffer = bufferPairs; bufferInfos[1].offset = 0; bufferInfos[1].range = pairsBufferSize;
    bufferInfos[2].buffer = bufferNumPairs; bufferInfos[2].offset = 0; bufferInfos[2].range = sizeof(int32_t);

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

    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = computeFamily;

    VkCommandPool commandPool;
    VK_CHECK(vkCreateCommandPool(device, &cmdPoolInfo, NULL, &commandPool));

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

    // Verify Output
    vkMapMemory(device, memoryNet, 0, netBufferSize, 0, &data);
    netData = (int32_t*)data;
    
    int successCount = 0;
    for (int i = 0; i < numPairs; i++) {
        int idx = i * 4;
        int expected = 10 + i;
        if (netData[idx + 3] == expected) {
            successCount++;
        } else {
            if (i < 10) {
                printf("[GPU DISPATCH] Error at pair %d: Expected %d, got %d\n", i, expected, netData[idx + 3]);
            }
        }
    }
    vkUnmapMemory(device, memoryNet);

    printf("[GPU DISPATCH] Processed %d pairs. Success: %d / %d\n", numPairs, successCount, numPairs);

    vkDestroyCommandPool(device, commandPool, NULL);
    vkDestroyPipeline(device, computePipeline, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
    
    vkDestroyBuffer(device, bufferNet, NULL); vkFreeMemory(device, memoryNet, NULL);
    vkDestroyBuffer(device, bufferPairs, NULL); vkFreeMemory(device, memoryPairs, NULL);
    vkDestroyBuffer(device, bufferNumPairs, NULL); vkFreeMemory(device, memoryNumPairs, NULL);

    vkDestroyShaderModule(device, computeShaderModule, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    free(devices);
    free(queueFamilies);
    free(shaderCode);
}
