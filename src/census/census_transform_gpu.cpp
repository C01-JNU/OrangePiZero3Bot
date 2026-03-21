#ifdef WITH_VULKAN

#include "census/census_transform.h"
#include "utils/logger.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unistd.h>
#include <limits.h>
#include <libgen.h>
#include <cmath>

namespace stereo_depth::census {

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

struct CensusTransform::GpuResources {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex = 0;

    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffers[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkFence fences[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    int currentFrame = 0;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;

    VkImage inputImage = VK_NULL_HANDLE;
    VkDeviceMemory inputImageMemory = VK_NULL_HANDLE;
    VkImageView inputImageView = VK_NULL_HANDLE;
    VkImage outputImage = VK_NULL_HANDLE;
    VkDeviceMemory outputImageMemory = VK_NULL_HANDLE;
    VkImageView outputImageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    bool layoutInitialized = false;

    VkBuffer stagingBuffers[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory stagingMemories[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    size_t stagingSize = 0;
    int currentStaging = 0;

    // 权重表 uniform buffer
    VkBuffer weightBuffer = VK_NULL_HANDLE;
    VkDeviceMemory weightMemory = VK_NULL_HANDLE;
    size_t weightSize = 0;
    int currentBilateralD = 0;      // 用于检测参数变化

    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;

    ~GpuResources();

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }

    static bool checkResult(VkResult result, const char* msg) {
        if (result != VK_SUCCESS) {
            LOG_ERROR("Vulkan error: {} -> {}", msg, static_cast<int>(result));
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
                           "Create shader module");
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        }
        throw std::runtime_error("Failed to find suitable memory type");
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
        if (!checkResult(vkCreateImage(device, &imageInfo, nullptr, &image), "Create image"))
            return false;

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!checkResult(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory), "Allocate image memory"))
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
        return checkResult(vkCreateImageView(device, &viewInfo, nullptr, &imageView), "Create image view");
    }

    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer,
                      VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!checkResult(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), "Create buffer"))
            return false;

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!checkResult(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory), "Allocate buffer memory"))
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

        VkImageMemoryBarrier outputBarrier = inputBarrier;
        outputBarrier.image = outputImage;

        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &inputBarrier);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &outputBarrier);
    }

    void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image,
                           uint32_t width, uint32_t height) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }

    void copyImageToBuffer(VkCommandBuffer cmd, VkImage image, VkBuffer buffer,
                           uint32_t width, uint32_t height) {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
    }

    bool dispatchCompute(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                         int win_w, int win_h, int wg_x, int wg_y,
                         int adaptive_thresh, int filter_type,
                         int bilateral_d, float sigma_color, float sigma_space) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        struct PushConstants {
            int ww, wh, iw, ih;
            int thresh;
            int ftype;
            int d;
            float scolor;
            float sspace;
        } pc = {win_w, win_h, (int)width, (int)height, adaptive_thresh, filter_type, bilateral_d, sigma_color, sigma_space};
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConstants), &pc);

        uint32_t groupX = (width + wg_x - 1) / wg_x;
        uint32_t groupY = (height + wg_y - 1) / wg_y;
        vkCmdDispatch(cmd, groupX, groupY, 1);
        return true;
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
            if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

CensusTransform::GpuResources::~GpuResources() { cleanup(); }

CensusTransform::~CensusTransform() {
    delete m_gpu;
}

bool CensusTransform::initGPU() {
    LOG_INFO("Initializing Vulkan for Census");
    m_gpu = new GpuResources();

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "CensusTransform";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    if (!GpuResources::checkResult(vkCreateInstance(&createInfo, nullptr, &m_gpu->instance), "Create instance")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_gpu->instance, &deviceCount, nullptr);
    if (deviceCount == 0) { LOG_ERROR("No Vulkan devices"); delete m_gpu; m_gpu = nullptr; return false; }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_gpu->instance, &deviceCount, devices.data());

    int selected = -1;
    for (uint32_t i = 0; i < deviceCount; ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        LOG_INFO("Device {}: {} (type: {})", i, props.deviceName, (int)props.deviceType);
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
    LOG_INFO("Selected device: {}", props.deviceName);

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
    if (!found) { LOG_ERROR("No compute queue"); delete m_gpu; m_gpu = nullptr; return false; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = m_gpu->queueFamilyIndex;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (!GpuResources::checkResult(vkCreateDevice(m_gpu->physicalDevice, &dci, nullptr, &m_gpu->device), "Create device")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    vkGetDeviceQueue(m_gpu->device, m_gpu->queueFamilyIndex, 0, &m_gpu->queue);

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = m_gpu->queueFamilyIndex;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (!GpuResources::checkResult(vkCreateCommandPool(m_gpu->device, &cpci, nullptr, &m_gpu->commandPool), "Create command pool")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = m_gpu->commandPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 3;
    if (!GpuResources::checkResult(vkAllocateCommandBuffers(m_gpu->device, &cbai, m_gpu->commandBuffers), "Allocate command buffers")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < 3; ++i) {
        if (!GpuResources::checkResult(vkCreateFence(m_gpu->device, &fci, nullptr, &m_gpu->fences[i]), "Create fence")) {
            delete m_gpu;
            m_gpu = nullptr;
            return false;
        }
    }

    VkSamplerCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter = VK_FILTER_NEAREST;
    sci.minFilter = VK_FILTER_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!GpuResources::checkResult(vkCreateSampler(m_gpu->device, &sci, nullptr, &m_gpu->sampler), "Create sampler")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    // 描述符集布局：binding0 输入图像，binding1 输出图像，binding2 权重表
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

    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = (uint32_t)bindings.size();
    dslci.pBindings = bindings.data();
    if (!GpuResources::checkResult(vkCreateDescriptorSetLayout(m_gpu->device, &dslci, nullptr, &m_gpu->descriptorSetLayout), "Create descriptor set layout")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    // 描述符池
    std::array<VkDescriptorPoolSize, 3> psizes{};
    psizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psizes[0].descriptorCount = 1;
    psizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    psizes[1].descriptorCount = 1;
    psizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    psizes[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.poolSizeCount = (uint32_t)psizes.size();
    dpci.pPoolSizes = psizes.data();
    dpci.maxSets = 1;
    if (!GpuResources::checkResult(vkCreateDescriptorPool(m_gpu->device, &dpci, nullptr, &m_gpu->descriptorPool), "Create descriptor pool")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = m_gpu->descriptorPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &m_gpu->descriptorSetLayout;
    if (!GpuResources::checkResult(vkAllocateDescriptorSets(m_gpu->device, &dsai, &m_gpu->descriptorSet), "Allocate descriptor set")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    // 创建管线布局
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(int)*7 + sizeof(float)*2; // 7个int + 2个float
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &m_gpu->descriptorSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (!GpuResources::checkResult(vkCreatePipelineLayout(m_gpu->device, &plci, nullptr, &m_gpu->pipelineLayout), "Create pipeline layout")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    // 加载着色器
    std::string exeDir = getExeDir();
    std::string shaderPath = exeDir + "/shaders/census.comp.spv";
    std::vector<char> shaderCode;
    try {
        shaderCode = GpuResources::readFile(shaderPath);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to read shader: {}", e.what());
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }
    if (!m_gpu->createShaderModule(shaderCode)) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = m_gpu->shaderModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = m_gpu->pipelineLayout;
    if (!GpuResources::checkResult(vkCreateComputePipelines(m_gpu->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_gpu->pipeline), "Create compute pipeline")) {
        delete m_gpu;
        m_gpu = nullptr;
        return false;
    }

    m_gpu->stagingSize = 0;
    LOG_INFO("Vulkan initialized");
    return true;
}

// 辅助函数：计算双边滤波权重表（固定数组大小 961，实际填充 bilateralD*bilateralD 个，其余置0）
static void computeWeightTables(float* spatialWeights, float* colorWeights,
                                int bilateral_d,
                                float sigmaSpace, float sigmaColor) {
    int total = bilateral_d * bilateral_d;
    float invSpaceVar = 1.0f / (2.0f * sigmaSpace * sigmaSpace);
    float invColorVar = 1.0f / (2.0f * sigmaColor * sigmaColor);
    int idx = 0;
    for (int dy = -bilateral_d/2; dy <= bilateral_d/2; ++dy) {
        for (int dx = -bilateral_d/2; dx <= bilateral_d/2; ++dx) {
            float dist2 = dx*dx + dy*dy;
            spatialWeights[idx++] = exp(-dist2 * invSpaceVar);
        }
    }
    // 其余空间权重（若 total < 961）保持为0，着色器不会访问
    for (int d = 0; d < 256; ++d) {
        colorWeights[d] = exp(-d*d * invColorVar);
    }
}

bool CensusTransform::computeGPU(const cv::Mat& src, cv::Mat& dst) {
    if (!m_gpu || !m_gpu->device) {
        LOG_ERROR("GPU not initialized");
        return false;
    }

    uint32_t w = src.cols, h = src.rows;
    size_t imageSize = w * h;
    size_t outputSize = w * h * sizeof(uint16_t);

    if (m_gpu->imageWidth != w || m_gpu->imageHeight != h) {
        if (m_gpu->inputImage) vkDestroyImage(m_gpu->device, m_gpu->inputImage, nullptr);
        if (m_gpu->inputImageMemory) vkFreeMemory(m_gpu->device, m_gpu->inputImageMemory, nullptr);
        if (m_gpu->outputImage) vkDestroyImage(m_gpu->device, m_gpu->outputImage, nullptr);
        if (m_gpu->outputImageMemory) vkFreeMemory(m_gpu->device, m_gpu->outputImageMemory, nullptr);
        if (m_gpu->inputImageView) vkDestroyImageView(m_gpu->device, m_gpu->inputImageView, nullptr);
        if (m_gpu->outputImageView) vkDestroyImageView(m_gpu->device, m_gpu->outputImageView, nullptr);
        for (int i = 0; i < 3; ++i) {
            if (m_gpu->stagingBuffers[i]) vkDestroyBuffer(m_gpu->device, m_gpu->stagingBuffers[i], nullptr);
            if (m_gpu->stagingMemories[i]) vkFreeMemory(m_gpu->device, m_gpu->stagingMemories[i], nullptr);
        }

        if (!m_gpu->createImage(w, h, VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                m_gpu->inputImage, m_gpu->inputImageMemory)) return false;
        if (!m_gpu->createImage(w, h, VK_FORMAT_R16_UINT, VK_IMAGE_TILING_OPTIMAL,
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                m_gpu->outputImage, m_gpu->outputImageMemory)) return false;
        if (!m_gpu->createImageView(m_gpu->inputImage, VK_FORMAT_R8_UNORM, m_gpu->inputImageView)) return false;
        if (!m_gpu->createImageView(m_gpu->outputImage, VK_FORMAT_R16_UINT, m_gpu->outputImageView)) return false;

        m_gpu->stagingSize = std::max(imageSize, outputSize);
        for (int i = 0; i < 3; ++i) {
            if (!m_gpu->createBuffer(m_gpu->stagingSize,
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     m_gpu->stagingBuffers[i], m_gpu->stagingMemories[i])) return false;
        }

        m_gpu->imageWidth = w;
        m_gpu->imageHeight = h;
        m_gpu->layoutInitialized = false;
    }

    // 更新权重表（如果双边滤波参数变化）
    if (m_gpu_filter_type == "bilateral") {
        static int lastD = -1;
        static float lastSigmaColor = -1, lastSigmaSpace = -1;
        if (lastD != m_gpu_bilateral_d || lastSigmaColor != m_gpu_bilateral_sigma_color || lastSigmaSpace != m_gpu_bilateral_sigma_space) {
            const int maxSize = 31 * 31; // 固定最大大小
            std::vector<float> spatialWeights(maxSize, 0.0f);
            std::vector<float> colorWeights(256);
            computeWeightTables(spatialWeights.data(), colorWeights.data(),
                                m_gpu_bilateral_d,
                                (float)m_gpu_bilateral_sigma_space,
                                (float)m_gpu_bilateral_sigma_color);
            size_t weightSize = maxSize * sizeof(float) + 256 * sizeof(float);
            // 确保缓冲区大小足够
            if (m_gpu->weightBuffer == VK_NULL_HANDLE || m_gpu->weightSize < weightSize) {
                if (m_gpu->weightBuffer) vkDestroyBuffer(m_gpu->device, m_gpu->weightBuffer, nullptr);
                if (m_gpu->weightMemory) vkFreeMemory(m_gpu->device, m_gpu->weightMemory, nullptr);
                if (!m_gpu->createBuffer(weightSize,
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         m_gpu->weightBuffer, m_gpu->weightMemory)) {
                    LOG_ERROR("Failed to create weight buffer of size {}", weightSize);
                    return false;
                }
                m_gpu->weightSize = weightSize;
            }
            void* ptr;
            vkMapMemory(m_gpu->device, m_gpu->weightMemory, 0, weightSize, 0, &ptr);
            memcpy(ptr, spatialWeights.data(), maxSize * sizeof(float));
            memcpy((char*)ptr + maxSize * sizeof(float), colorWeights.data(), 256 * sizeof(float));
            vkUnmapMemory(m_gpu->device, m_gpu->weightMemory);
            lastD = m_gpu_bilateral_d;
            lastSigmaColor = m_gpu_bilateral_sigma_color;
            lastSigmaSpace = m_gpu_bilateral_sigma_space;
        }
    }

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
        VkDescriptorImageInfo outputInfo{};
        outputInfo.sampler = m_gpu->sampler;
        outputInfo.imageView = m_gpu->outputImageView;
        outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorBufferInfo weightInfo{};
        weightInfo.buffer = m_gpu->weightBuffer;
        weightInfo.offset = 0;
        weightInfo.range = m_gpu->weightSize;

        VkWriteDescriptorSet writes[3];
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->descriptorSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inputInfo, nullptr, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->descriptorSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_gpu->descriptorSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &weightInfo, nullptr};
        vkUpdateDescriptorSets(m_gpu->device, 3, writes, 0, nullptr);
    }

    int frameIdx = m_gpu->currentFrame;
    int nextFrameIdx = (frameIdx + 1) % 3;
    int stagingIdx = m_gpu->currentStaging;
    int nextStagingIdx = (stagingIdx + 1) % 3;

    vkWaitForFences(m_gpu->device, 1, &m_gpu->fences[frameIdx], VK_TRUE, UINT64_MAX);
    vkResetFences(m_gpu->device, 1, &m_gpu->fences[frameIdx]);

    void* mapped;
    vkMapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx], 0, imageSize, 0, &mapped);
    std::memcpy(mapped, src.data, imageSize);
    vkUnmapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx]);

    vkResetCommandBuffer(m_gpu->commandBuffers[frameIdx], 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_gpu->commandBuffers[frameIdx], &beginInfo);

    m_gpu->copyBufferToImage(m_gpu->commandBuffers[frameIdx], m_gpu->stagingBuffers[stagingIdx],
                             m_gpu->inputImage, w, h);

    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(m_gpu->commandBuffers[frameIdx], VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    int filter_type = 0;
    if (m_gpu_filter_type == "median") filter_type = 1;
    else if (m_gpu_filter_type == "bilateral") filter_type = 2;
    else filter_type = 0;

    if (!m_gpu->dispatchCompute(m_gpu->commandBuffers[frameIdx], w, h,
                                m_win_width, m_win_height,
                                m_workgroup_x, m_workgroup_y,
                                m_adaptive_threshold,
                                filter_type,
                                m_gpu_bilateral_d,
                                (float)m_gpu_bilateral_sigma_color,
                                (float)m_gpu_bilateral_sigma_space)) {
        return false;
    }

    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(m_gpu->commandBuffers[frameIdx], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);

    m_gpu->copyImageToBuffer(m_gpu->commandBuffers[frameIdx], m_gpu->outputImage,
                             m_gpu->stagingBuffers[stagingIdx], w, h);

    vkEndCommandBuffer(m_gpu->commandBuffers[frameIdx]);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_gpu->commandBuffers[frameIdx];

    if (!GpuResources::checkResult(vkQueueSubmit(m_gpu->queue, 1, &submitInfo,
                                                  m_gpu->fences[frameIdx]),
                                   "Submit compute commands")) {
        return false;
    }

    m_gpu->currentFrame = nextFrameIdx;
    m_gpu->currentStaging = nextStagingIdx;

    vkWaitForFences(m_gpu->device, 1, &m_gpu->fences[frameIdx], VK_TRUE, UINT64_MAX);

    vkMapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx], 0, outputSize, 0, &mapped);
    dst.create(h, w, CV_16U);
    std::memcpy(dst.data, mapped, outputSize);
    vkUnmapMemory(m_gpu->device, m_gpu->stagingMemories[stagingIdx]);

    return true;
}

} // namespace stereo_depth::census

#endif // WITH_VULKAN
