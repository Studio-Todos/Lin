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

void pic_gpu_dispatch(int32_t opcode, int32_t lhs, int32_t rhs, int32_t* out) {
    printf("[GPU DISPATCH] Launching kernel for opcode %d with lhs=%d, rhs=%d\n", opcode, lhs, rhs);

    FILE *f = fopen("linc_out.spv", "rb");
    if (!f) {
        printf("[GPU DISPATCH] Error: SPIR-V module linc_out.spv not found\n");
        return;
    }
    fseek(f, 0, SEEK_END);
    long fileSize = ftell(f);
    rewind(f);
    uint32_t* shaderCode = (uint32_t*)malloc(fileSize);
    fread(shaderCode, 1, fileSize, f);
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
    
    // Enable validation layers if requested/available
    // const char* validationLayers[] = {"VK_LAYER_KHRONOS_validation"};
    // createInfo.enabledLayerCount = 1;
    // createInfo.ppEnabledLayerNames = validationLayers;

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
    VkPhysicalDevice physicalDevice = devices[0]; // Just pick the first one

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

    // Create 4 buffers: opcode, lhs, rhs, out
    VkBuffer buffer0, buffer1, buffer2, buffer3;
    VkDeviceMemory memory0, memory1, memory2, memory3;
    
    createBuffer(device, physicalDevice, sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer0, &memory0);
    createBuffer(device, physicalDevice, sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer1, &memory1);
    createBuffer(device, physicalDevice, sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer2, &memory2);
    createBuffer(device, physicalDevice, sizeof(int32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buffer3, &memory3);

    void* data;
    vkMapMemory(device, memory0, 0, sizeof(int32_t), 0, &data);
    memcpy(data, &opcode, sizeof(int32_t));
    vkUnmapMemory(device, memory0);

    vkMapMemory(device, memory1, 0, sizeof(int32_t), 0, &data);
    memcpy(data, &lhs, sizeof(int32_t));
    vkUnmapMemory(device, memory1);

    vkMapMemory(device, memory2, 0, sizeof(int32_t), 0, &data);
    memcpy(data, &rhs, sizeof(int32_t));
    vkUnmapMemory(device, memory2);

    int32_t zeroOut = 0;
    vkMapMemory(device, memory3, 0, sizeof(int32_t), 0, &data);
    memcpy(data, &zeroOut, sizeof(int32_t));
    vkUnmapMemory(device, memory3);

    VkDescriptorSetLayoutBinding bindings[4];
    for (int i = 0; i < 4; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = NULL;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 4;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &layoutInfo, NULL, &descriptorSetLayout));

    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 4;

    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;

    VkDescriptorPool descriptorPool;
    VK_CHECK(vkCreateDescriptorPool(device, &poolInfo, NULL, &descriptorPool));

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));

    VkDescriptorBufferInfo bufferInfos[4];
    bufferInfos[0].buffer = buffer0; bufferInfos[0].offset = 0; bufferInfos[0].range = sizeof(int32_t);
    bufferInfos[1].buffer = buffer1; bufferInfos[1].offset = 0; bufferInfos[1].range = sizeof(int32_t);
    bufferInfos[2].buffer = buffer2; bufferInfos[2].offset = 0; bufferInfos[2].range = sizeof(int32_t);
    bufferInfos[3].buffer = buffer3; bufferInfos[3].offset = 0; bufferInfos[3].range = sizeof(int32_t);

    VkWriteDescriptorSet descriptorWrites[4];
    for (int i = 0; i < 4; i++) {
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
    vkUpdateDescriptorSets(device, 4, descriptorWrites, 0, NULL);

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
    pipelineInfo.stage.pName = "pic_kernel"; // MUST match the shader entry point
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

    vkCmdDispatch(commandBuffer, 1, 1, 1);
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK(vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(computeQueue));

    vkMapMemory(device, memory3, 0, sizeof(int32_t), 0, &data);
    int32_t result;
    memcpy(&result, data, sizeof(int32_t));
    vkUnmapMemory(device, memory3);

    printf("[GPU DISPATCH] Result from GPU: %d\n", result);
    if (out) *out = result;

    vkDestroyCommandPool(device, commandPool, NULL);
    vkDestroyPipeline(device, computePipeline, NULL);
    vkDestroyPipelineLayout(device, pipelineLayout, NULL);
    vkDestroyDescriptorPool(device, descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, NULL);
    
    vkDestroyBuffer(device, buffer0, NULL); vkFreeMemory(device, memory0, NULL);
    vkDestroyBuffer(device, buffer1, NULL); vkFreeMemory(device, memory1, NULL);
    vkDestroyBuffer(device, buffer2, NULL); vkFreeMemory(device, memory2, NULL);
    vkDestroyBuffer(device, buffer3, NULL); vkFreeMemory(device, memory3, NULL);

    vkDestroyShaderModule(device, computeShaderModule, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    free(devices);
    free(queueFamilies);
    free(shaderCode);
}
