// Copyright (C) 2026 C01-JNU
// SPDX-License-Identifier: GPL-3.0-only
//
// This file is part of FishTotem.
//
// FishTotem is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// FishTotem is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with FishTotem. If not, see <https://www.gnu.org/licenses/>.


#include "preprocess/gpu_bilateral.h"
#include "utils/logger.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <cstring>
#include <cmath>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>

namespace stereo_depth::preprocess {

static std::string getExeDir() {
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        result[count] = '\0';
        char* dir = dirname(result);
        return std::string(dir);
    }
    return ".";
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("无法打开文件: " + filename);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

struct GpuBilateralFilter::Impl {
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;

    VkCommandBuffer commandBuffers[2];
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    VkImage inputImage = VK_NULL_HANDLE;
    VkDeviceMemory inputImageMemory = VK_NULL_HANDLE;
    VkImageView inputImageView = VK_NULL_HANDLE;
    VkImage outputImage = VK_NULL_HANDLE;
    VkDeviceMemory outputImageMemory = VK_NULL_HANDLE;
    VkImageView outputImageView = VK_NULL_HANDLE;

    VkSampler sampler = VK_NULL_HANDLE;

    VkBuffer weightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory weightMemory = VK_NULL_HANDLE;

    VkBuffer stagingBuffers[2];
    VkDeviceMemory stagingMemories[2];
    size_t stagingSize = 0;
    int currentIndex = 0;

    int diameter = 0, radius = 0;
    float sigmaColor = 0.0f, sigmaSpace = 0.0f;
    int curWidth = 0, curHeight = 0;
    bool initialized = false;

    ~Impl() { cleanup(); }

    bool checkResult(VkResult result, const char* msg) {
        if (result != VK_SUCCESS) {
            LOG_ERROR("Vulkan 错误: {} -> {}", msg, static_cast<int>(result));
            return false;
        }
        return true;
    }

    bool createShaderModule(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        return checkResult(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule),
                           "创建着色器模块");
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("找不到合适的内存类型");
    }

    bool createImage(uint32_t width, uint32_t height, VkFormat format,
                     VkImageTiling tiling, VkImageUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkImage& image,
                     VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent = {width, height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!checkResult(vkCreateImage(device, &imageInfo, nullptr, &image), "创建图像")) {
            return false;
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!checkResult(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory), "分配图像内存")) {
            return false;
        }
        vkBindImageMemory(device, image, imageMemory, 0);
        return true;
    }

    bool createImageView(VkImage image, VkFormat format, VkImageView& imageView) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        return checkResult(vkCreateImageView(device, &viewInfo, nullptr, &imageView), "创建图像视图");
    }

    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!checkResult(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), "创建缓冲区")) {
            return false;
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!checkResult(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory), "分配缓冲区内存")) {
            return false;
        }
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
        return true;
    }

    void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = (newLayout == VK_IMAGE_LAYOUT_GENERAL) ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT : 0;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }

    void copyImageToBuffer(VkCommandBuffer cmd, VkImage image, VkBuffer buffer, uint32_t width, uint32_t height) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
    }

    void computeWeightTables(float* spatialWeights, float* colorWeights) {
        float invSpaceVar = 1.0f / (2.0f * sigmaSpace * sigmaSpace);
        float invColorVar = 1.0f / (2.0f * sigmaColor * sigmaColor);
        int idx = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                float dist2 = dx*dx + dy*dy;
                spatialWeights[idx++] = expf(-dist2 * invSpaceVar);
            }
        }
        for (int d = 0; d < 256; ++d) {
            colorWeights[d] = expf(-d*d * invColorVar);
        }
    }

    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            for (int i = 0; i < 2; ++i) {
                if (commandBuffers[i]) vkFreeCommandBuffers(device, commandPool, 1, &commandBuffers[i]);
                if (stagingBuffers[i]) vkDestroyBuffer(device, stagingBuffers[i], nullptr);
                if (stagingMemories[i]) vkFreeMemory(device, stagingMemories[i], nullptr);
            }
            if (weightBuffer) vkDestroyBuffer(device, weightBuffer, nullptr);
            if (weightMemory) vkFreeMemory(device, weightMemory, nullptr);
            if (sampler) vkDestroySampler(device, sampler, nullptr);
            if (outputImageView) vkDestroyImageView(device, outputImageView, nullptr);
            if (inputImageView) vkDestroyImageView(device, inputImageView, nullptr);
            if (outputImage) vkDestroyImage(device, outputImage, nullptr);
            if (inputImage) vkDestroyImage(device, inputImage, nullptr);
            if (outputImageMemory) vkFreeMemory(device, outputImageMemory, nullptr);
            if (inputImageMemory) vkFreeMemory(device, inputImageMemory, nullptr);
            if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
            if (shaderModule) vkDestroyShaderModule(device, shaderModule, nullptr);
        }
    }
};

GpuBilateralFilter::GpuBilateralFilter() : m_impl(std::make_unique<Impl>()) {}
GpuBilateralFilter::~GpuBilateralFilter() = default;

bool GpuBilateralFilter::init(int diameter, float sigmaColor, float sigmaSpace,
                              VkPhysicalDevice physicalDevice, VkDevice device,
                              VkQueue queue, VkCommandPool commandPool) {
    auto& impl = *m_impl;
    impl.physicalDevice = physicalDevice;
    impl.device = device;
    impl.queue = queue;
    impl.commandPool = commandPool;
    impl.diameter = diameter;
    impl.radius = diameter / 2;
    impl.sigmaColor = sigmaColor;
    impl.sigmaSpace = sigmaSpace;

    // 分配 2 个命令缓冲
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = impl.commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;
    if (!impl.checkResult(vkAllocateCommandBuffers(impl.device, &cbai, impl.commandBuffers), "分配命令缓冲")) {
        return false;
    }

    // 创建采样器
    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!impl.checkResult(vkCreateSampler(impl.device, &sci, nullptr, &impl.sampler), "创建采样器")) {
        return false;
    }

    // 加载着色器
    std::string exeDir = getExeDir();
    std::string shaderPath = exeDir + "/shaders/bilateral.comp.spv";
    std::vector<char> shaderCode;
    try {
        shaderCode = readFile(shaderPath);
    } catch (const std::exception& e) {
        LOG_ERROR("读取着色器失败: {}", e.what());
        return false;
    }
    if (!impl.createShaderModule(shaderCode)) {
        return false;
    }

    // 描述符集布局
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (!impl.checkResult(vkCreateDescriptorSetLayout(impl.device, &layoutInfo, nullptr, &impl.descriptorSetLayout),
                          "创建描述符集布局")) {
        return false;
    }

    // 描述符池
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 1;
    if (!impl.checkResult(vkCreateDescriptorPool(impl.device, &poolInfo, nullptr, &impl.descriptorPool),
                          "创建描述符池")) {
        return false;
    }

    // 分配描述符集
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = impl.descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &impl.descriptorSetLayout;
    if (!impl.checkResult(vkAllocateDescriptorSets(impl.device, &allocInfo, &impl.descriptorSet),
                          "分配描述符集")) {
        return false;
    }

    // 管线布局
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(int) * 2 + sizeof(float) * 3;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &impl.descriptorSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pushRange;
    if (!impl.checkResult(vkCreatePipelineLayout(impl.device, &plci, nullptr, &impl.pipelineLayout),
                          "创建管线布局")) {
        return false;
    }

    // 计算管线
    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = impl.shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = impl.pipelineLayout;
    if (!impl.checkResult(vkCreateComputePipelines(impl.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
                                                   &impl.pipeline), "创建计算管线")) {
        return false;
    }

    // 预创建权重表缓冲区
    constexpr size_t MAX_SPATIAL = 961;
    constexpr size_t COLOR_SIZE = 256;
    size_t weightSize = (MAX_SPATIAL + COLOR_SIZE) * sizeof(float);
    if (!impl.createBuffer(weightSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           impl.weightBuffer, impl.weightMemory)) {
        return false;
    }

    std::vector<float> spatialWeights(MAX_SPATIAL, 0.0f);
    std::vector<float> colorWeights(COLOR_SIZE);
    impl.computeWeightTables(spatialWeights.data(), colorWeights.data());
    void* ptr;
    vkMapMemory(impl.device, impl.weightMemory, 0, weightSize, 0, &ptr);
    memcpy(ptr, spatialWeights.data(), MAX_SPATIAL * sizeof(float));
    memcpy(static_cast<char*>(ptr) + MAX_SPATIAL * sizeof(float), colorWeights.data(), COLOR_SIZE * sizeof(float));
    vkUnmapMemory(impl.device, impl.weightMemory);

    impl.initialized = true;
    LOG_INFO("双边滤波 GPU 模块初始化成功");
    return true;
}

bool GpuBilateralFilter::process(const cv::Mat& inputBgr, cv::Mat& outputBgr) {
    auto& impl = *m_impl;
    if (!impl.initialized) {
        LOG_ERROR("双边滤波未初始化");
        return false;
    }
    int w = inputBgr.cols, h = inputBgr.rows;

    // 动态创建图像和缓冲区（如果尺寸变化或首次调用）
    if (impl.curWidth != w || impl.curHeight != h) {
        // 销毁旧资源
        if (impl.inputImage) vkDestroyImage(impl.device, impl.inputImage, nullptr);
        if (impl.inputImageMemory) vkFreeMemory(impl.device, impl.inputImageMemory, nullptr);
        if (impl.outputImage) vkDestroyImage(impl.device, impl.outputImage, nullptr);
        if (impl.outputImageMemory) vkFreeMemory(impl.device, impl.outputImageMemory, nullptr);
        if (impl.inputImageView) vkDestroyImageView(impl.device, impl.inputImageView, nullptr);
        if (impl.outputImageView) vkDestroyImageView(impl.device, impl.outputImageView, nullptr);
        for (int i = 0; i < 2; ++i) {
            if (impl.stagingBuffers[i]) vkDestroyBuffer(impl.device, impl.stagingBuffers[i], nullptr);
            if (impl.stagingMemories[i]) vkFreeMemory(impl.device, impl.stagingMemories[i], nullptr);
        }

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        if (!impl.createImage(w, h, format, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              impl.inputImage, impl.inputImageMemory)) {
            return false;
        }
        if (!impl.createImage(w, h, format, VK_IMAGE_TILING_OPTIMAL,
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              impl.outputImage, impl.outputImageMemory)) {
            return false;
        }
        if (!impl.createImageView(impl.inputImage, format, impl.inputImageView) ||
            !impl.createImageView(impl.outputImage, format, impl.outputImageView)) {
            return false;
        }

        size_t imageSize = w * h * 4;
        impl.stagingSize = imageSize;
        for (int i = 0; i < 2; ++i) {
            if (!impl.createBuffer(impl.stagingSize,
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                   impl.stagingBuffers[i], impl.stagingMemories[i])) {
                return false;
            }
        }

        // 初始化图像布局
        VkCommandBuffer cmd = impl.commandBuffers[0];
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
        impl.transitionImageLayout(cmd, impl.inputImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        impl.transitionImageLayout(cmd, impl.outputImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(impl.queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(impl.queue);
        vkResetCommandBuffer(cmd, 0);

        // 更新描述符集
        VkDescriptorImageInfo inputInfo{};
        inputInfo.sampler = impl.sampler;
        inputInfo.imageView = impl.inputImageView;
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo outputInfo{};
        outputInfo.sampler = impl.sampler;
        outputInfo.imageView = impl.outputImageView;
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo weightInfo{};
        weightInfo.buffer = impl.weightBuffer;
        weightInfo.offset = 0;
        weightInfo.range = (961 + 256) * sizeof(float);

        VkWriteDescriptorSet writes[3];
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, impl.descriptorSet, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inputInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, impl.descriptorSet, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, impl.descriptorSet, 2, 0, 1,
                     VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &weightInfo, nullptr};
        vkUpdateDescriptorSets(impl.device, 3, writes, 0, nullptr);

        impl.curWidth = w;
        impl.curHeight = h;
        impl.currentIndex = 0;
    }

    size_t imageSize = w * h * 4;

    int idx = impl.currentIndex;
    int nextIdx = (idx + 1) % 2;

    vkQueueWaitIdle(impl.queue);

    // 上传 BGR -> RGBA
    void* mapped;
    vkMapMemory(impl.device, impl.stagingMemories[idx], 0, imageSize, 0, &mapped);
    uint8_t* dst = static_cast<uint8_t*>(mapped);
    const uint8_t* src = inputBgr.data;
    for (size_t i = 0; i < w * h; ++i) {
        dst[4*i + 0] = src[3*i + 2];
        dst[4*i + 1] = src[3*i + 1];
        dst[4*i + 2] = src[3*i + 0];
        dst[4*i + 3] = 255;
    }
    vkUnmapMemory(impl.device, impl.stagingMemories[idx]);

    VkCommandBuffer cmd = impl.commandBuffers[idx];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    impl.copyBufferToImage(cmd, impl.stagingBuffers[idx], impl.inputImage, w, h);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl.pipelineLayout,
                            0, 1, &impl.descriptorSet, 0, nullptr);

    struct PushConstants { int w; int h; int radius; float sigmaColor; float sigmaSpace; } pc = {
        w, h, impl.radius, impl.sigmaColor, impl.sigmaSpace
    };
    vkCmdPushConstants(cmd, impl.pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groupX = (w + 15) / 16;
    uint32_t groupY = (h + 15) / 16;
    vkCmdDispatch(cmd, groupX, groupY, 1);

    // 直接拷贝回 buffer
    impl.copyImageToBuffer(cmd, impl.outputImage, impl.stagingBuffers[idx], w, h);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    if (!impl.checkResult(vkQueueSubmit(impl.queue, 1, &submitInfo, VK_NULL_HANDLE), "提交计算")) {
        return false;
    }

    vkQueueWaitIdle(impl.queue);

    // 读回结果 RGBA -> BGR
    vkMapMemory(impl.device, impl.stagingMemories[idx], 0, imageSize, 0, &mapped);
    outputBgr.create(h, w, CV_8UC3);
    uint8_t* outPtr = outputBgr.data;
    const uint8_t* inPtr = static_cast<const uint8_t*>(mapped);
    for (size_t i = 0; i < w * h; ++i) {
        outPtr[3*i + 0] = inPtr[4*i + 2];
        outPtr[3*i + 1] = inPtr[4*i + 1];
        outPtr[3*i + 2] = inPtr[4*i + 0];
    }
    vkUnmapMemory(impl.device, impl.stagingMemories[idx]);

    impl.currentIndex = nextIdx;

    return true;
}

void* GpuBilateralFilter::getOutputImageView() const {
    if (!m_impl->initialized) return nullptr;
    return static_cast<void*>(m_impl->outputImageView);
}
int GpuBilateralFilter::getWidth() const { return m_impl->curWidth; }
int GpuBilateralFilter::getHeight() const { return m_impl->curHeight; }

} // namespace stereo_depth::preprocess
