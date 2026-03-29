#ifdef WITH_VULKAN

#include "preprocess/preprocess.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <algorithm>
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

static void computeBilateralWeights(float* spatialWeights, float* colorWeights,
                                    int diameter, float sigmaSpace, float sigmaColor) {
    int radius = diameter / 2;
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

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("无法打开文件: " + filename);
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

struct Preprocess::GpuResources {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffers[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkFence fences[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    int currentFrame = 0;

    VkDescriptorSetLayout bilateralLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout censusLayout = VK_NULL_HANDLE;
    VkPipelineLayout bilateralPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout censusPipelineLayout = VK_NULL_HANDLE;
    VkPipeline bilateralPipeline = VK_NULL_HANDLE;
    VkPipeline censusPipeline = VK_NULL_HANDLE;
    VkShaderModule bilateralShader = VK_NULL_HANDLE;
    VkShaderModule censusShader = VK_NULL_HANDLE;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet bilateralSet = VK_NULL_HANDLE;
    VkDescriptorSet censusSet = VK_NULL_HANDLE;

    VkImage inputImage = VK_NULL_HANDLE;
    VkDeviceMemory inputImageMemory = VK_NULL_HANDLE;
    VkImageView inputImageView = VK_NULL_HANDLE;
    VkImage filteredImage = VK_NULL_HANDLE;
    VkDeviceMemory filteredImageMemory = VK_NULL_HANDLE;
    VkImageView filteredImageView = VK_NULL_HANDLE;
    VkImage censusImage = VK_NULL_HANDLE;
    VkDeviceMemory censusImageMemory = VK_NULL_HANDLE;
    VkImageView censusImageView = VK_NULL_HANDLE;

    VkSampler sampler = VK_NULL_HANDLE;

    VkBuffer weightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory weightMemory = VK_NULL_HANDLE;
    size_t weightSize = 0;

    VkBuffer stagingBuffers[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory stagingMemories[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    size_t stagingSize = 0;
    int currentStaging = 0;

    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    bool layoutInitialized = false;

    int bilateral_d = 9;
    float bilateral_sigma_color = 50.0f;
    float bilateral_sigma_space = 9.0f;
    int bilateral_radius = 4;
    int bilateral_total = 81;

    ~GpuResources();

    bool checkResult(VkResult result, const char* msg) {
        if (result != VK_SUCCESS) {
            LOG_ERROR("Vulkan 错误: {} -> {}", msg, static_cast<int>(result));
            return false;
        }
        return true;
    }

    bool createShaderModule(const std::vector<char>& code, VkShaderModule& module) {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        return checkResult(vkCreateShaderModule(device, &createInfo, nullptr, &module),
                           "创建着色器模块");
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
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
        if (!checkResult(vkCreateImage(device, &imageInfo, nullptr, &image), "创建图像"))
            return false;

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!checkResult(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory), "分配图像内存"))
            return false;
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
        if (!checkResult(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), "创建缓冲区"))
            return false;

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!checkResult(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory), "分配缓冲区内存"))
            return false;
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
        return true;
    }

    void initImageLayout(VkCommandBuffer cmd) {
        VkImageMemoryBarrier inputBarrier{};
        inputBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        inputBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        inputBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        inputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        inputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        inputBarrier.image = inputImage;
        inputBarrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        inputBarrier.srcAccessMask = 0;
        inputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

        VkImageMemoryBarrier filteredBarrier = inputBarrier;
        filteredBarrier.image = filteredImage;
        VkImageMemoryBarrier censusBarrier = inputBarrier;
        censusBarrier.image = censusImage;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &inputBarrier);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &filteredBarrier);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &censusBarrier);
    }

    void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                           uint32_t width, uint32_t height, VkImageLayout layout) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, buffer, image, layout, 1, &region);
    }

    void copyImageToBuffer(VkCommandBuffer cmd, VkImage image, VkBuffer buffer,
                           uint32_t width, uint32_t height, VkImageLayout layout) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, image, layout, buffer, 1, &region);
    }

    void recordBilateralCommands(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                 int wg_x, int wg_y) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = inputImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, bilateralPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                bilateralPipelineLayout, 0, 1, &bilateralSet, 0, nullptr);

        struct BilateralPushConstants {
            int imageWidth;
            int imageHeight;
            int radius;
            float sigmaColor;
            float sigmaSpace;
        } pc = {static_cast<int>(width), static_cast<int>(height),
                bilateral_radius, bilateral_sigma_color, bilateral_sigma_space};
        vkCmdPushConstants(cmd, bilateralPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t groupX = (width + wg_x - 1) / wg_x;
        uint32_t groupY = (height + wg_y - 1) / wg_y;
        vkCmdDispatch(cmd, groupX, groupY, 1);
    }

    void recordCensusCommands(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                              int win_w, int win_h, int thresh, int wg_x, int wg_y) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = filteredImage;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, censusPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                censusPipelineLayout, 0, 1, &censusSet, 0, nullptr);

        struct CensusPushConstants {
            int windowWidth;
            int windowHeight;
            int imageWidth;
            int imageHeight;
            int adaptiveThreshold;
        } pc = {win_w, win_h, static_cast<int>(width), static_cast<int>(height), thresh};
        vkCmdPushConstants(cmd, censusPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);

        uint32_t groupX = (width + wg_x - 1) / wg_x;
        uint32_t groupY = (height + wg_y - 1) / wg_y;
        vkCmdDispatch(cmd, groupX, groupY, 1);
    }

    void cleanup() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            for (int i = 0; i < 3; ++i) {
                if (fences[i]) vkDestroyFence(device, fences[i], nullptr);
                if (commandBuffers[i]) vkFreeCommandBuffers(device, commandPool, 1, &commandBuffers[i]);
                if (stagingBuffers[i]) vkDestroyBuffer(device, stagingBuffers[i], nullptr);
                if (stagingMemories[i]) vkFreeMemory(device, stagingMemories[i], nullptr);
            }
            if (weightBuffer) vkDestroyBuffer(device, weightBuffer, nullptr);
            if (weightMemory) vkFreeMemory(device, weightMemory, nullptr);
            if (sampler) vkDestroySampler(device, sampler, nullptr);
            if (censusImageView) vkDestroyImageView(device, censusImageView, nullptr);
            if (filteredImageView) vkDestroyImageView(device, filteredImageView, nullptr);
            if (inputImageView) vkDestroyImageView(device, inputImageView, nullptr);
            if (censusImage) vkDestroyImage(device, censusImage, nullptr);
            if (filteredImage) vkDestroyImage(device, filteredImage, nullptr);
            if (inputImage) vkDestroyImage(device, inputImage, nullptr);
            if (censusImageMemory) vkFreeMemory(device, censusImageMemory, nullptr);
            if (filteredImageMemory) vkFreeMemory(device, filteredImageMemory, nullptr);
            if (inputImageMemory) vkFreeMemory(device, inputImageMemory, nullptr);
            if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            if (bilateralSet) vkFreeDescriptorSets(device, descriptorPool, 1, &bilateralSet);
            if (censusSet) vkFreeDescriptorSets(device, descriptorPool, 1, &censusSet);
            if (bilateralPipeline) vkDestroyPipeline(device, bilateralPipeline, nullptr);
            if (censusPipeline) vkDestroyPipeline(device, censusPipeline, nullptr);
            if (bilateralPipelineLayout) vkDestroyPipelineLayout(device, bilateralPipelineLayout, nullptr);
            if (censusPipelineLayout) vkDestroyPipelineLayout(device, censusPipelineLayout, nullptr);
            if (bilateralLayout) vkDestroyDescriptorSetLayout(device, bilateralLayout, nullptr);
            if (censusLayout) vkDestroyDescriptorSetLayout(device, censusLayout, nullptr);
            if (bilateralShader) vkDestroyShaderModule(device, bilateralShader, nullptr);
            if (censusShader) vkDestroyShaderModule(device, censusShader, nullptr);
            if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

Preprocess::GpuResources::~GpuResources() { cleanup(); }

bool Preprocess::initGPU() {
    LOG_INFO("初始化 Vulkan 前处理模块");

    m_gpu = new GpuResources();
    if (!m_gpu) {
        LOG_ERROR("创建 GpuResources 失败");
        return false;
    }
    LOG_INFO("GpuResources 创建成功, m_gpu={}", reinterpret_cast<void*>(m_gpu));

    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    m_gpu->bilateral_d = cfg.get<int>("preprocess.denoise.bilateral_d", 9);
    if (m_gpu->bilateral_d > 31) {
        LOG_WARN("双边滤波直径 {} 超过最大支持31，将使用31", m_gpu->bilateral_d);
        m_gpu->bilateral_d = 31;
    }
    if (m_gpu->bilateral_d % 2 == 0) {
        m_gpu->bilateral_d++;
        LOG_WARN("双边滤波直径调整为奇数: {}", m_gpu->bilateral_d);
    }
    m_gpu->bilateral_sigma_color = static_cast<float>(cfg.get<double>("preprocess.denoise.bilateral_sigma_color", 50.0));
    m_gpu->bilateral_sigma_space = static_cast<float>(cfg.get<double>("preprocess.denoise.bilateral_sigma_space", 9.0));
    m_gpu->bilateral_radius = m_gpu->bilateral_d / 2;
    m_gpu->bilateral_total = m_gpu->bilateral_d * m_gpu->bilateral_d;
    LOG_INFO("双边滤波参数: d={}, sigma_color={}, sigma_space={}",
             m_gpu->bilateral_d, m_gpu->bilateral_sigma_color, m_gpu->bilateral_sigma_space);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Preprocess";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    if (!m_gpu->checkResult(vkCreateInstance(&instanceInfo, nullptr, &m_gpu->instance), "创建 Vulkan 实例"))
        return false;

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_gpu->instance, &deviceCount, nullptr);
    if (deviceCount == 0) { LOG_ERROR("未找到 Vulkan 设备"); return false; }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_gpu->instance, &deviceCount, devices.data());

    int selected = -1;
    for (uint32_t i = 0; i < deviceCount; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        LOG_INFO("设备 {}: {} (类型: {})", i, props.deviceName, static_cast<int>(props.deviceType));
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            selected = i;
            break;
        }
    }
    if (selected == -1) selected = 0;
    m_gpu->physicalDevice = devices[selected];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(m_gpu->physicalDevice, &props);
    LOG_INFO("选中设备: {}", props.deviceName);

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_gpu->physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_gpu->physicalDevice, &qfCount, qfs.data());
    bool found = false;
    for (uint32_t i = 0; i < qfCount; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            m_gpu->queueFamilyIndex = i;
            found = true;
            break;
        }
    }
    if (!found) { LOG_ERROR("未找到支持计算的队列族"); return false; }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = m_gpu->queueFamilyIndex;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queuePriority;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (!m_gpu->checkResult(vkCreateDevice(m_gpu->physicalDevice, &dci, nullptr, &m_gpu->device), "创建设备"))
        return false;

    vkGetDeviceQueue(m_gpu->device, m_gpu->queueFamilyIndex, 0, &m_gpu->queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = m_gpu->queueFamilyIndex;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (!m_gpu->checkResult(vkCreateCommandPool(m_gpu->device, &cpci, nullptr, &m_gpu->commandPool), "创建命令池"))
        return false;

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = m_gpu->commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 3;
    if (!m_gpu->checkResult(vkAllocateCommandBuffers(m_gpu->device, &cbai, m_gpu->commandBuffers), "分配命令缓冲"))
        return false;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < 3; ++i) {
        if (!m_gpu->checkResult(vkCreateFence(m_gpu->device, &fci, nullptr, &m_gpu->fences[i]), "创建栅栏"))
            return false;
    }

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!m_gpu->checkResult(vkCreateSampler(m_gpu->device, &sci, nullptr, &m_gpu->sampler), "创建采样器"))
        return false;

    // 加载着色器
    std::string exeDir = getExeDir();
    std::string bilateralShaderPath = exeDir + "/shaders/bilateral.comp.spv";
    std::string censusShaderPath = exeDir + "/shaders/census_adaptive.comp.spv";
    std::vector<char> bilateralCode, censusCode;
    try {
        bilateralCode = readFile(bilateralShaderPath);
        censusCode = readFile(censusShaderPath);
    } catch (const std::exception& e) {
        LOG_ERROR("读取着色器文件失败: {}", e.what());
        return false;
    }
    if (!m_gpu->createShaderModule(bilateralCode, m_gpu->bilateralShader)) return false;
    if (!m_gpu->createShaderModule(censusCode, m_gpu->censusShader)) return false;

    // 创建双边滤波描述符集布局
    std::array<VkDescriptorSetLayoutBinding, 3> bilateralBindings{};
    bilateralBindings[0].binding = 0;
    bilateralBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bilateralBindings[0].descriptorCount = 1;
    bilateralBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bilateralBindings[1].binding = 1;
    bilateralBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bilateralBindings[1].descriptorCount = 1;
    bilateralBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bilateralBindings[2].binding = 2;
    bilateralBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bilateralBindings[2].descriptorCount = 1;
    bilateralBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo bilateralLayoutInfo{};
    bilateralLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    bilateralLayoutInfo.bindingCount = static_cast<uint32_t>(bilateralBindings.size());
    bilateralLayoutInfo.pBindings = bilateralBindings.data();
    if (!m_gpu->checkResult(vkCreateDescriptorSetLayout(m_gpu->device, &bilateralLayoutInfo, nullptr, &m_gpu->bilateralLayout), "创建双边滤波描述符集布局"))
        return false;

    // 创建 Census 描述符集布局
    std::array<VkDescriptorSetLayoutBinding, 2> censusBindings{};
    censusBindings[0].binding = 0;
    censusBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    censusBindings[0].descriptorCount = 1;
    censusBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    censusBindings[1].binding = 1;
    censusBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    censusBindings[1].descriptorCount = 1;
    censusBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo censusLayoutInfo{};
    censusLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    censusLayoutInfo.bindingCount = static_cast<uint32_t>(censusBindings.size());
    censusLayoutInfo.pBindings = censusBindings.data();
    if (!m_gpu->checkResult(vkCreateDescriptorSetLayout(m_gpu->device, &censusLayoutInfo, nullptr, &m_gpu->censusLayout), "创建 Census 描述符集布局"))
        return false;

    // 创建描述符池
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 2;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 2;
    if (!m_gpu->checkResult(vkCreateDescriptorPool(m_gpu->device, &poolInfo, nullptr, &m_gpu->descriptorPool), "创建描述符池"))
        return false;

    // 分配描述符集
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_gpu->descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_gpu->bilateralLayout;
    if (!m_gpu->checkResult(vkAllocateDescriptorSets(m_gpu->device, &allocInfo, &m_gpu->bilateralSet), "分配双边滤波描述符集"))
        return false;
    allocInfo.pSetLayouts = &m_gpu->censusLayout;
    if (!m_gpu->checkResult(vkAllocateDescriptorSets(m_gpu->device, &allocInfo, &m_gpu->censusSet), "分配 Census 描述符集"))
        return false;

    // 创建管线布局
    VkPushConstantRange bilateralPush{};
    bilateralPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bilateralPush.offset = 0;
    bilateralPush.size = sizeof(int)*2 + sizeof(float)*3;

    VkPipelineLayoutCreateInfo bilateralPlci{};
    bilateralPlci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    bilateralPlci.setLayoutCount = 1;
    bilateralPlci.pSetLayouts = &m_gpu->bilateralLayout;
    bilateralPlci.pushConstantRangeCount = 1;
    bilateralPlci.pPushConstantRanges = &bilateralPush;
    if (!m_gpu->checkResult(vkCreatePipelineLayout(m_gpu->device, &bilateralPlci, nullptr, &m_gpu->bilateralPipelineLayout), "创建双边滤波管线布局"))
        return false;

    VkPushConstantRange censusPush{};
    censusPush.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    censusPush.offset = 0;
    censusPush.size = sizeof(int)*5;

    VkPipelineLayoutCreateInfo censusPlci{};
    censusPlci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    censusPlci.setLayoutCount = 1;
    censusPlci.pSetLayouts = &m_gpu->censusLayout;
    censusPlci.pushConstantRangeCount = 1;
    censusPlci.pPushConstantRanges = &censusPush;
    if (!m_gpu->checkResult(vkCreatePipelineLayout(m_gpu->device, &censusPlci, nullptr, &m_gpu->censusPipelineLayout), "创建 Census 管线布局"))
        return false;

    // 创建计算管线
    VkComputePipelineCreateInfo bilateralPipelineInfo{};
    bilateralPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    bilateralPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    bilateralPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    bilateralPipelineInfo.stage.module = m_gpu->bilateralShader;
    bilateralPipelineInfo.stage.pName = "main";
    bilateralPipelineInfo.layout = m_gpu->bilateralPipelineLayout;
    if (!m_gpu->checkResult(vkCreateComputePipelines(m_gpu->device, VK_NULL_HANDLE, 1, &bilateralPipelineInfo, nullptr, &m_gpu->bilateralPipeline), "创建双边滤波管线"))
        return false;

    VkComputePipelineCreateInfo censusPipelineInfo{};
    censusPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    censusPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    censusPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    censusPipelineInfo.stage.module = m_gpu->censusShader;
    censusPipelineInfo.stage.pName = "main";
    censusPipelineInfo.layout = m_gpu->censusPipelineLayout;
    if (!m_gpu->checkResult(vkCreateComputePipelines(m_gpu->device, VK_NULL_HANDLE, 1, &censusPipelineInfo, nullptr, &m_gpu->censusPipeline), "创建 Census 管线"))
        return false;

    LOG_INFO("Vulkan 前处理模块初始化完成");
    return true;
}

bool Preprocess::processGPU(const cv::Mat& left, const cv::Mat& right,
                            cv::Mat& left_census, cv::Mat& right_census) {
    LOG_INFO("processGPU: 进入函数");
    if (!m_gpu) {
        LOG_ERROR("processGPU: m_gpu 为空");
        return false;
    }
    LOG_INFO("processGPU: m_gpu 地址 = {}", reinterpret_cast<void*>(m_gpu));
    if (!m_gpu->device) {
        LOG_ERROR("processGPU: m_gpu->device 为空");
        return false;
    }
    LOG_INFO("processGPU: 图像尺寸 {}x{}", left.cols, left.rows);

    uint32_t w = left.cols, h = left.rows;
    if (left.size() != right.size()) {
        LOG_ERROR("左右图像尺寸不一致");
        return false;
    }
    if (left.type() != CV_8UC3 || right.type() != CV_8UC3) {
        LOG_ERROR("GPU 模式要求输入图像为 CV_8UC3 彩色图");
        return false;
    }

    size_t imageSize = w * h * 3;
    size_t outputSize = w * h * sizeof(uint16_t);

    int wgx = m_census.getWorkgroupX();
    int wgy = m_census.getWorkgroupY();

    // 检查图像尺寸变化，重建资源
    if (m_gpu->imageWidth != w || m_gpu->imageHeight != h) {
        LOG_INFO("图像尺寸变化，重建 GPU 资源 ({}x{} -> {}x{})",
                 m_gpu->imageWidth, m_gpu->imageHeight, w, h);
        // 清理旧资源
        if (m_gpu->inputImage) vkDestroyImage(m_gpu->device, m_gpu->inputImage, nullptr);
        if (m_gpu->inputImageMemory) vkFreeMemory(m_gpu->device, m_gpu->inputImageMemory, nullptr);
        if (m_gpu->filteredImage) vkDestroyImage(m_gpu->device, m_gpu->filteredImage, nullptr);
        if (m_gpu->filteredImageMemory) vkFreeMemory(m_gpu->device, m_gpu->filteredImageMemory, nullptr);
        if (m_gpu->censusImage) vkDestroyImage(m_gpu->device, m_gpu->censusImage, nullptr);
        if (m_gpu->censusImageMemory) vkFreeMemory(m_gpu->device, m_gpu->censusImageMemory, nullptr);
        if (m_gpu->inputImageView) vkDestroyImageView(m_gpu->device, m_gpu->inputImageView, nullptr);
        if (m_gpu->filteredImageView) vkDestroyImageView(m_gpu->device, m_gpu->filteredImageView, nullptr);
        if (m_gpu->censusImageView) vkDestroyImageView(m_gpu->device, m_gpu->censusImageView, nullptr);
        for (int i = 0; i < 3; ++i) {
            if (m_gpu->stagingBuffers[i]) vkDestroyBuffer(m_gpu->device, m_gpu->stagingBuffers[i], nullptr);
            if (m_gpu->stagingMemories[i]) vkFreeMemory(m_gpu->device, m_gpu->stagingMemories[i], nullptr);
        }

        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
        if (!m_gpu->createImage(w, h, format, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                m_gpu->inputImage, m_gpu->inputImageMemory)) return false;
        if (!m_gpu->createImage(w, h, format, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                m_gpu->filteredImage, m_gpu->filteredImageMemory)) return false;
        if (!m_gpu->createImage(w, h, VK_FORMAT_R16_UINT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                m_gpu->censusImage, m_gpu->censusImageMemory)) return false;
        if (!m_gpu->createImageView(m_gpu->inputImage, format, m_gpu->inputImageView)) return false;
        if (!m_gpu->createImageView(m_gpu->filteredImage, format, m_gpu->filteredImageView)) return false;
        if (!m_gpu->createImageView(m_gpu->censusImage, VK_FORMAT_R16_UINT, m_gpu->censusImageView)) return false;

        m_gpu->stagingSize = std::max(imageSize, outputSize);
        for (int i = 0; i < 3; ++i) {
            if (!m_gpu->createBuffer(m_gpu->stagingSize,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     m_gpu->stagingBuffers[i], m_gpu->stagingMemories[i])) return false;
        }

        m_gpu->currentFrame = 0;
        m_gpu->currentStaging = 0;
        m_gpu->imageWidth = w;
        m_gpu->imageHeight = h;
        m_gpu->layoutInitialized = false;
    }

    // 确保权重表已创建
    if (m_gpu->weightBuffer == VK_NULL_HANDLE) {
        constexpr size_t MAX_SPATIAL = 961;
        constexpr size_t COLOR_SIZE = 256;
        size_t weightSize = (MAX_SPATIAL + COLOR_SIZE) * sizeof(float);
        if (!m_gpu->createBuffer(weightSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 m_gpu->weightBuffer, m_gpu->weightMemory)) {
            LOG_ERROR("创建权重表缓冲区失败");
            return false;
        }
        m_gpu->weightSize = weightSize;

        std::vector<float> spatialWeights(MAX_SPATIAL, 0.0f);
        std::vector<float> colorWeights(COLOR_SIZE);
        computeBilateralWeights(spatialWeights.data(), colorWeights.data(),
                                m_gpu->bilateral_d,
                                m_gpu->bilateral_sigma_space,
                                m_gpu->bilateral_sigma_color);

        void* ptr;
        vkMapMemory(m_gpu->device, m_gpu->weightMemory, 0, weightSize, 0, &ptr);
        memcpy(ptr, spatialWeights.data(), MAX_SPATIAL * sizeof(float));
        memcpy(static_cast<char*>(ptr) + MAX_SPATIAL * sizeof(float),
               colorWeights.data(), COLOR_SIZE * sizeof(float));
        vkUnmapMemory(m_gpu->device, m_gpu->weightMemory);

        LOG_INFO("权重表已创建 (直径={})", m_gpu->bilateral_d);
    }

    // 初始化图像布局
    if (!m_gpu->layoutInitialized) {
        VkCommandBuffer cmd = m_gpu->commandBuffers[0];
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);
        m_gpu->initImageLayout(cmd);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(m_gpu->queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_gpu->queue);
        vkResetCommandBuffer(cmd, 0);
        m_gpu->layoutInitialized = true;

        // 绑定描述符集
        VkDescriptorImageInfo inputInfo{};
        inputInfo.sampler = m_gpu->sampler;
        inputInfo.imageView = m_gpu->inputImageView;
        inputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo filteredInfo{};
        filteredInfo.sampler = m_gpu->sampler;
        filteredInfo.imageView = m_gpu->filteredImageView;
        filteredInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo censusInfo{};
        censusInfo.sampler = m_gpu->sampler;
        censusInfo.imageView = m_gpu->censusImageView;
        censusInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet writes[3];
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->bilateralSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inputInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->bilateralSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &filteredInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(m_gpu->device, 2, writes, 0, nullptr);

        VkDescriptorBufferInfo weightInfo{};
        weightInfo.buffer = m_gpu->weightBuffer;
        weightInfo.offset = 0;
        weightInfo.range = m_gpu->weightSize;
        VkWriteDescriptorSet weightWrite = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->bilateralSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &weightInfo, nullptr};
        vkUpdateDescriptorSets(m_gpu->device, 1, &weightWrite, 0, nullptr);

        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->censusSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &filteredInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->censusSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &censusInfo, nullptr, nullptr};
        vkUpdateDescriptorSets(m_gpu->device, 2, writes, 0, nullptr);
    }

    // 处理单张图像
    auto processOne = [&](const cv::Mat& src, cv::Mat& dst) -> bool {
        uint32_t w = src.cols, h = src.rows;
        size_t imageSize = w * h * 3;
        size_t outputSize = w * h * sizeof(uint16_t);

        int frameIdx = m_gpu->currentFrame;
        int nextFrameIdx = (frameIdx + 1) % 3;
        int stagingIdx = m_gpu->currentStaging;
        int nextStagingIdx = (stagingIdx + 1) % 3;

        vkWaitForFences(m_gpu->device, 1, &m_gpu->fences[frameIdx], VK_TRUE, UINT64_MAX);
        vkResetFences(m_gpu->device, 1, &m_gpu->fences[frameIdx]);

        void* mapped;
        vkMapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx], 0, imageSize, 0, &mapped);
        uint8_t* dst_ptr = static_cast<uint8_t*>(mapped);
        const uint8_t* src_ptr = src.data;
        for (size_t i = 0; i < w * h; ++i) {
            dst_ptr[4*i + 0] = src_ptr[3*i + 0];
            dst_ptr[4*i + 1] = src_ptr[3*i + 1];
            dst_ptr[4*i + 2] = src_ptr[3*i + 2];
            dst_ptr[4*i + 3] = 255;
        }
        vkUnmapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx]);

        vkResetCommandBuffer(m_gpu->commandBuffers[frameIdx], 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(m_gpu->commandBuffers[frameIdx], &beginInfo);

        m_gpu->copyBufferToImage(m_gpu->commandBuffers[frameIdx],
                                 m_gpu->stagingBuffers[stagingIdx],
                                 m_gpu->inputImage, w, h, VK_IMAGE_LAYOUT_GENERAL);

        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(m_gpu->commandBuffers[frameIdx], VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        m_gpu->recordBilateralCommands(m_gpu->commandBuffers[frameIdx], w, h, wgx, wgy);

        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(m_gpu->commandBuffers[frameIdx], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        m_gpu->recordCensusCommands(m_gpu->commandBuffers[frameIdx], w, h,
                                    m_census.getWindowWidth(), m_census.getWindowHeight(),
                                    m_census.getAdaptiveThreshold(), wgx, wgy);

        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(m_gpu->commandBuffers[frameIdx], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

        m_gpu->copyImageToBuffer(m_gpu->commandBuffers[frameIdx],
                                 m_gpu->censusImage,
                                 m_gpu->stagingBuffers[stagingIdx],
                                 w, h, VK_IMAGE_LAYOUT_GENERAL);

        vkEndCommandBuffer(m_gpu->commandBuffers[frameIdx]);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_gpu->commandBuffers[frameIdx];
        if (!m_gpu->checkResult(vkQueueSubmit(m_gpu->queue, 1, &submitInfo, m_gpu->fences[frameIdx]),
                                "提交计算命令")) {
            return false;
        }

        m_gpu->currentFrame = nextFrameIdx;
        m_gpu->currentStaging = nextStagingIdx;

        vkWaitForFences(m_gpu->device, 1, &m_gpu->fences[frameIdx], VK_TRUE, UINT64_MAX);

        vkMapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx], 0, outputSize, 0, &mapped);
        dst.create(h, w, CV_16U);
        memcpy(dst.data, mapped, outputSize);
        vkUnmapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx]);

        return true;
    };

    if (!processOne(left, left_census)) return false;
    if (!processOne(right, right_census)) return false;

    return true;
}

Preprocess::~Preprocess() {
    delete m_gpu;
}

} // namespace stereo_depth::preprocess

#endif // WITH_VULKAN
