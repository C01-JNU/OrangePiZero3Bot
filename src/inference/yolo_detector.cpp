#include "inference/yolo_detector.h"
#include "utils/logger.hpp"
#include "utils/config.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace stereo_depth::inference {

YOLODetector::YOLODetector() = default;

YOLODetector::~YOLODetector() {
    if (m_interpreter && m_session) {
        m_interpreter->releaseSession(m_session);
    }
}

bool YOLODetector::initFromConfig() {
    auto& cfg = utils::ConfigManager::getInstance().getConfig();
    YOLOConfig config;
    config.model_path = cfg.get<std::string>("inference.model.path", "models/yolo26n.mnn");
    config.input_width = cfg.get<int>("inference.model.input_width", 0);
    config.input_height = cfg.get<int>("inference.model.input_height", 0);
    config.confidence_threshold = cfg.get<float>("inference.confidence_threshold", 0.25f);
    config.nms_threshold = cfg.get<float>("inference.nms_threshold", 0.45f);
    config.num_classes = cfg.get<int>("inference.num_classes", 80);
    config.use_gpu = cfg.get<bool>("inference.use_gpu", false);
    config.letterbox = cfg.get<bool>("inference.letterbox", true);
    LOG_INFO("YOLO config: model={}, input=({},{}), conf={}, nms={}, letterbox={}",
             config.model_path, config.input_width, config.input_height,
             config.confidence_threshold, config.nms_threshold, config.letterbox);
    return init(config);
}

bool YOLODetector::init(const YOLOConfig& config) {
    m_config = config;
    if (m_config.input_width > 0 && m_config.input_height > 0) {
        // 固定尺寸，直接加载
        if (!loadModel(m_config.input_width, m_config.input_height)) return false;
        m_input_size_fixed = true;
    } else {
        // 自动尺寸：等待第一帧
        LOG_INFO("Input size not set, will auto-detect from first frame");
        m_input_size_fixed = false;
    }
    m_initialized = true;
    LOG_INFO("YOLODetector initialized (auto-size mode={})", !m_input_size_fixed);
    return true;
}

bool YOLODetector::loadModel(int input_width, int input_height) {
    if (m_interpreter) {
        if (m_session) m_interpreter->releaseSession(m_session);
        m_interpreter.reset();
    }
    m_interpreter.reset(MNN::Interpreter::createFromFile(m_config.model_path.c_str()));
    if (!m_interpreter) {
        LOG_ERROR("Failed to load model: {}", m_config.model_path);
        return false;
    }
    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.type = MNN_FORWARD_CPU;
    scheduleConfig.numThread = 4;
    m_session = m_interpreter->createSession(scheduleConfig);
    if (!m_session) {
        LOG_ERROR("Failed to create session");
        return false;
    }
    MNN::Tensor* inputTensor = m_interpreter->getSessionInput(m_session, nullptr);
    if (!inputTensor) {
        LOG_ERROR("No input tensor");
        return false;
    }
    std::vector<int> expectedShape = {1, 3, input_height, input_width};
    auto actualShape = inputTensor->shape();
    if (actualShape.size() != 4 || actualShape[1] != 3 ||
        actualShape[2] != input_height || actualShape[3] != input_width) {
        LOG_INFO("Resizing input tensor to {}x{}", input_width, input_height);
        m_interpreter->resizeTensor(inputTensor, expectedShape);
        m_interpreter->resizeSession(m_session);
    }
    LOG_INFO("Model loaded with input size {}x{}", input_width, input_height);
    return true;
}

bool YOLODetector::preprocess(const cv::Mat& image, MNN::Tensor* input_tensor,
                              float& scale_x, float& scale_y,
                              int& pad_left, int& pad_top) {
    int model_w = m_config.input_width;
    int model_h = m_config.input_height;
    int img_w = image.cols;
    int img_h = image.rows;

    cv::Mat resized;
    if (m_config.letterbox) {
        // 计算缩放比例和灰边
        float scale = std::min(static_cast<float>(model_w) / img_w,
                               static_cast<float>(model_h) / img_h);
        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);
        pad_left = (model_w - new_w) / 2;
        pad_top = (model_h - new_h) / 2;
        scale_x = scale;
        scale_y = scale;

        cv::Mat scaled;
        cv::resize(image, scaled, cv::Size(new_w, new_h));
        resized = cv::Mat(model_h, model_w, CV_8UC3, cv::Scalar(114, 114, 114));
        scaled.copyTo(resized(cv::Rect(pad_left, pad_top, new_w, new_h)));
    } else {
        // 直接拉伸
        cv::resize(image, resized, cv::Size(model_w, model_h));
        scale_x = static_cast<float>(model_w) / img_w;
        scale_y = static_cast<float>(model_h) / img_h;
        pad_left = pad_top = 0;
    }

    auto input_ptr = input_tensor->host<float>();
    if (!input_ptr) return false;
    int stride = model_w * model_h;
    for (int h = 0; h < model_h; ++h) {
        for (int w = 0; w < model_w; ++w) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(h, w);
            input_ptr[0 * stride + h * model_w + w] = pixel[2] / 255.0f;
            input_ptr[1 * stride + h * model_w + w] = pixel[1] / 255.0f;
            input_ptr[2 * stride + h * model_w + w] = pixel[0] / 255.0f;
        }
    }
    return true;
}

bool YOLODetector::postprocess(const std::vector<MNN::Tensor*>& output_tensors,
                               std::vector<DetectionBox>& results,
                               float scale_x, float scale_y,
                               int pad_left, int pad_top) {
    results.clear();
    if (output_tensors.empty()) return false;
    auto* output = output_tensors[0];
    auto shape = output->shape();
    // 假设输出 [1, num_boxes, 6]
    if (shape.size() == 3 && shape[2] == 6) {
        int num_boxes = shape[1];
        auto* data = output->host<float>();
        for (int i = 0; i < num_boxes; ++i) {
            float* box = data + i * 6;
            float conf = box[4];
            if (conf < m_config.confidence_threshold) continue;
            int class_id = static_cast<int>(box[5]);
            if (class_id < 0 || class_id >= m_config.num_classes) continue;
            DetectionBox det;
            // 将模型输出坐标（相对于模型输入）映射回原图坐标
            float x1 = (box[0] * m_config.input_width - pad_left) / scale_x / m_config.input_width;
            float y1 = (box[1] * m_config.input_height - pad_top) / scale_y / m_config.input_height;
            float x2 = (box[2] * m_config.input_width - pad_left) / scale_x / m_config.input_width;
            float y2 = (box[3] * m_config.input_height - pad_top) / scale_y / m_config.input_height;
            det.x1 = std::clamp(x1, 0.0f, 1.0f);
            det.y1 = std::clamp(y1, 0.0f, 1.0f);
            det.x2 = std::clamp(x2, 0.0f, 1.0f);
            det.y2 = std::clamp(y2, 0.0f, 1.0f);
            det.confidence = conf;
            det.class_id = class_id;
            det.class_name = "class_" + std::to_string(class_id);
            results.push_back(det);
        }
    } else {
        LOG_WARN("Unsupported output shape");
    }
    applyNMS(results, m_config.nms_threshold);
    return true;
}

void YOLODetector::applyNMS(std::vector<DetectionBox>& boxes, float iou_threshold) {
    if (boxes.empty()) return;
    std::sort(boxes.begin(), boxes.end(),
              [](auto& a, auto& b) { return a.confidence > b.confidence; });
    std::vector<bool> keep(boxes.size(), true);
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!keep[i]) continue;
        for (size_t j = i+1; j < boxes.size(); ++j) {
            if (!keep[j]) continue;
            float ix1 = std::max(boxes[i].x1, boxes[j].x1);
            float iy1 = std::max(boxes[i].y1, boxes[j].y1);
            float ix2 = std::min(boxes[i].x2, boxes[j].x2);
            float iy2 = std::min(boxes[i].y2, boxes[j].y2);
            if (ix2 > ix1 && iy2 > iy1) {
                float inter = (ix2-ix1)*(iy2-iy1);
                float area_i = (boxes[i].x2-boxes[i].x1)*(boxes[i].y2-boxes[i].y1);
                float area_j = (boxes[j].x2-boxes[j].x1)*(boxes[j].y2-boxes[j].y1);
                float iou = inter / (area_i+area_j-inter);
                if (iou > iou_threshold) keep[j] = false;
            }
        }
    }
    std::vector<DetectionBox> filtered;
    for (size_t i = 0; i < boxes.size(); ++i)
        if (keep[i]) filtered.push_back(boxes[i]);
    boxes.swap(filtered);
}

bool YOLODetector::detect(const cv::Mat& image, std::vector<DetectionBox>& results) {
    if (!m_initialized) return false;
    if (image.empty()) {
        LOG_ERROR("Empty image");
        return false;
    }
    // 自动确定输入尺寸（第一帧）
    if (!m_input_size_fixed) {
        int auto_w = m_config.input_width;
        int auto_h = m_config.input_height;
        if (auto_w <= 0 || auto_h <= 0) {
            // 默认取图像宽高中的最大值，并向上对齐到32的倍数（常见要求）
            auto_w = ((image.cols + 31) / 32) * 32;
            auto_h = ((image.rows + 31) / 32) * 32;
            LOG_INFO("Auto-set input size to {}x{} based on first frame ({}x{})",
                     auto_w, auto_h, image.cols, image.rows);
        }
        if (!loadModel(auto_w, auto_h)) return false;
        m_config.input_width = auto_w;
        m_config.input_height = auto_h;
        m_input_size_fixed = true;
    }
    MNN::Tensor* inputTensor = m_interpreter->getSessionInput(m_session, nullptr);
    if (!inputTensor) return false;
    float scale_x, scale_y;
    int pad_left, pad_top;
    if (!preprocess(image, inputTensor, scale_x, scale_y, pad_left, pad_top))
        return false;
    m_interpreter->runSession(m_session);
    MNN::Tensor* outputTensor = m_interpreter->getSessionOutput(m_session, nullptr);
    if (!outputTensor) return false;
    std::vector<MNN::Tensor*> outputs = {outputTensor};
    if (!postprocess(outputs, results, scale_x, scale_y, pad_left, pad_top))
        return false;
    LOG_DEBUG("Detected {} objects", results.size());
    return true;
}

bool YOLODetector::detectFromGpuImage(void* vkImageView, int width, int height,
                                      std::vector<DetectionBox>& results) {
    LOG_WARN("GPU inference not implemented yet");
    results.clear();
    return false;
}

void YOLODetector::setVulkanResources(void* device, void* queue, void* commandPool) {
    m_vk_device = device;
    m_vk_queue = queue;
    m_vk_command_pool = commandPool;
}

} // namespace stereo_depth::inference
