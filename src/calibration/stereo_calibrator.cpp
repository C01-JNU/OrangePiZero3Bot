#include "calibration/stereo_calibrator.hpp"
#include "utils/config.hpp"
#include "utils/logger.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <limits.h>

namespace stereo_depth {
namespace calibration {

static bool fileExists(const std::string& path) {
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

static std::string getAbsolutePath(const std::string& path) {
    if (path.empty() || path[0] == '/') return path;
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) return path;
    std::string absPath = std::string(cwd) + "/" + path;
    char resolved[PATH_MAX];
    if (realpath(absPath.c_str(), resolved) != nullptr)
        return std::string(resolved);
    return absPath;
}

static bool createDirectory(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        if (mkdir(path.c_str(), 0777) != 0) return false;
    }
    return true;
}

bool StereoCalibrator::loadConfiguration() {
    auto& config = utils::ConfigManager::getInstance().getConfig();

    int bw = config.get<int>("calibration.board_width", 1);
    int bh = config.get<int>("calibration.board_height", 1);
    if (bw <= 0 || bh <= 0) {
        LOG_ERROR("棋盘格尺寸配置错误: board_width={}, board_height={}", bw, bh);
        return false;
    }
    m_boardSize = cv::Size(bw, bh);

    m_squareSize = config.get<float>("calibration.square_size", 1.0f);
    if (m_squareSize <= 0) {
        LOG_ERROR("方格尺寸配置错误");
        return false;
    }

    int cw = config.get<int>("camera.width", 1);
    int ch = config.get<int>("camera.height", 1);
    m_imageSize = cv::Size(cw / 2, ch);
    if (m_imageSize.width <= 0 || m_imageSize.height <= 0) {
        LOG_ERROR("图像尺寸配置错误");
        return false;
    }

    std::string calibDirRel = config.get<std::string>("output.calibration_dir", "images/calibration");
    if (calibDirRel.empty()) {
        LOG_ERROR("未配置 output.calibration_dir");
        return false;
    }
    if (calibDirRel[0] == '/') {
        m_calibrationDir = calibDirRel;
    } else {
        if (!m_basePath.empty()) {
            m_calibrationDir = m_basePath + "/" + calibDirRel;
        } else {
            m_calibrationDir = calibDirRel;
        }
    }
    LOG_INFO("标定图像目录: {}", m_calibrationDir);

    std::string calibFileRel = config.get<std::string>("calibration.calibration_file", "calibration_results/stereo_calibration.yml");
    if (calibFileRel.empty()) {
        LOG_ERROR("未配置 calibration.calibration_file");
        return false;
    }
    size_t pos = calibFileRel.find_last_of('/');
    std::string outputDirRel;
    if (pos != std::string::npos)
        outputDirRel = calibFileRel.substr(0, pos);
    else
        outputDirRel = ".";
    if (outputDirRel[0] == '/') {
        m_outputDir = outputDirRel;
    } else {
        if (!m_basePath.empty()) {
            m_outputDir = m_basePath + "/" + outputDirRel;
        } else {
            m_outputDir = outputDirRel;
        }
    }
    if (!createDirectory(m_outputDir)) {
        LOG_ERROR("无法创建输出目录: {}", m_outputDir);
        return false;
    }
    LOG_INFO("输出目录: {}", m_outputDir);

    return true;
}

std::vector<std::pair<std::string, std::string>> StereoCalibrator::findCalibrationImagePairs() {
    std::vector<std::pair<std::string, std::string>> pairs;

    DIR* dir = opendir(m_calibrationDir.c_str());
    if (!dir) {
        LOG_ERROR("无法打开目录: {}", m_calibrationDir);
        return pairs;
    }

    std::vector<std::string> leftFiles;
    struct dirent* entry;
    while ((entry = readdir(dir))) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("_left") != std::string::npos &&
            (lower.find(".jpg") != std::string::npos ||
             lower.find(".png") != std::string::npos ||
             lower.find(".jpeg") != std::string::npos ||
             lower.find(".bmp") != std::string::npos)) {
            leftFiles.push_back(name);
        }
    }
    closedir(dir);
    std::sort(leftFiles.begin(), leftFiles.end());

    LOG_INFO("找到 {} 个左眼图像", leftFiles.size());

    for (const auto& leftName : leftFiles) {
        std::string base = leftName;
        size_t p = base.find("_left");
        if (p == std::string::npos) continue;
        std::string rightName = base.substr(0, p) + "_right" + base.substr(p + 5);
        std::string leftPath = m_calibrationDir + "/" + leftName;
        std::string rightPath = m_calibrationDir + "/" + rightName;

        if (fileExists(rightPath)) {
            pairs.emplace_back(leftPath, rightPath);
            LOG_DEBUG("匹配对: {} <-> {}", leftName, rightName);
        } else {
            LOG_WARN("找不到对应的右图: {}", rightName);
        }
    }

    LOG_INFO("共找到 {} 对有效图像", pairs.size());
    return pairs;
}

bool StereoCalibrator::detectChessboardCorners(const cv::Mat& image, std::vector<cv::Point2f>& corners) {
    bool found = cv::findChessboardCorners(image, m_boardSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    return found;
}

bool StereoCalibrator::performCalibration(
    const std::vector<std::vector<cv::Point3f>>& objectPoints,
    const std::vector<std::vector<cv::Point2f>>& imagePointsLeft,
    const std::vector<std::vector<cv::Point2f>>& imagePointsRight) {

    m_cameraMatrixLeft = cv::Mat::eye(3, 3, CV_64F);
    m_cameraMatrixRight = cv::Mat::eye(3, 3, CV_64F);
    m_distCoeffsLeft = cv::Mat::zeros(5, 1, CV_64F);
    m_distCoeffsRight = cv::Mat::zeros(5, 1, CV_64F);

    std::vector<cv::Mat> rvecsL, tvecsL, rvecsR, tvecsR;
    double rmsL = cv::calibrateCamera(objectPoints, imagePointsLeft, m_imageSize,
                                       m_cameraMatrixLeft, m_distCoeffsLeft, rvecsL, tvecsL);
    double rmsR = cv::calibrateCamera(objectPoints, imagePointsRight, m_imageSize,
                                       m_cameraMatrixRight, m_distCoeffsRight, rvecsR, tvecsR);
    LOG_DEBUG("单目标定 RMS: 左={:.6f}, 右={:.6f}", rmsL, rmsR);

    int flags = cv::CALIB_USE_INTRINSIC_GUESS;
    cv::TermCriteria term(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6);
    m_rmsError = cv::stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight,
                                      m_cameraMatrixLeft, m_distCoeffsLeft,
                                      m_cameraMatrixRight, m_distCoeffsRight,
                                      m_imageSize, m_rotationMatrix, m_translationVector,
                                      m_essentialMatrix, m_fundamentalMatrix,
                                      flags, term);

    LOG_INFO("双目标定 RMS = {:.6f}", m_rmsError);
    return true;
}

bool StereoCalibrator::calibrate() {
    LOG_INFO("开始立体相机标定");

    if (!loadConfiguration()) {
        LOG_ERROR("加载配置失败");
        return false;
    }

    auto imagePairs = findCalibrationImagePairs();
    if (imagePairs.empty()) {
        LOG_ERROR("没有找到任何标定图像对");
        return false;
    }

    std::vector<cv::Point3f> obj;
    for (int i = 0; i < m_boardSize.height; ++i) {
        for (int j = 0; j < m_boardSize.width; ++j) {
            obj.emplace_back(j * m_squareSize, i * m_squareSize, 0.0f);
        }
    }

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePointsLeft, imagePointsRight;
    int validPairs = 0;

    for (size_t i = 0; i < imagePairs.size(); ++i) {
        const auto& [leftPath, rightPath] = imagePairs[i];
        LOG_INFO("处理图像对 {}/{}: {}", i+1, imagePairs.size(), leftPath);

        cv::Mat leftImg = cv::imread(leftPath, cv::IMREAD_GRAYSCALE);
        cv::Mat rightImg = cv::imread(rightPath, cv::IMREAD_GRAYSCALE);
        if (leftImg.empty() || rightImg.empty()) {
            LOG_WARN("读取图像失败，跳过");
            continue;
        }

        if (leftImg.size() != m_imageSize || rightImg.size() != m_imageSize) {
            cv::resize(leftImg, leftImg, m_imageSize);
            cv::resize(rightImg, rightImg, m_imageSize);
        }

        std::vector<cv::Point2f> cornersLeft, cornersRight;
        bool leftOk = detectChessboardCorners(leftImg, cornersLeft);
        bool rightOk = detectChessboardCorners(rightImg, cornersRight);

        if (leftOk && rightOk) {
            cv::cornerSubPix(leftImg, cornersLeft, cv::Size(5,5), cv::Size(-1,-1),
                              cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            cv::cornerSubPix(rightImg, cornersRight, cv::Size(5,5), cv::Size(-1,-1),
                              cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
            objectPoints.push_back(obj);
            imagePointsLeft.push_back(cornersLeft);
            imagePointsRight.push_back(cornersRight);
            validPairs++;
        } else {
            LOG_WARN("角点检测失败，跳过");
        }
    }

    if (validPairs < 3) {
        LOG_ERROR("有效图像对太少，无法标定 (需要至少3对，实际 {})", validPairs);
        return false;
    }

    LOG_INFO("有效图像对数量: {}", validPairs);
    m_imagesUsed = validPairs;

    if (!performCalibration(objectPoints, imagePointsLeft, imagePointsRight)) {
        LOG_ERROR("标定失败");
        return false;
    }

    cv::Rect validRoi[2];
    cv::stereoRectify(m_cameraMatrixLeft, m_distCoeffsLeft,
                      m_cameraMatrixRight, m_distCoeffsRight,
                      m_imageSize, m_rotationMatrix, m_translationVector,
                      m_rectificationTransformLeft, m_rectificationTransformRight,
                      m_projectionMatrixLeft, m_projectionMatrixRight,
                      m_disparityToDepthMappingMatrix,
                      cv::CALIB_ZERO_DISPARITY, 1, m_imageSize,
                      &validRoi[0], &validRoi[1]);
    m_validRoiLeft = validRoi[0];
    m_validRoiRight = validRoi[1];

    cv::initUndistortRectifyMap(m_cameraMatrixLeft, m_distCoeffsLeft,
                                 m_rectificationTransformLeft, m_projectionMatrixLeft,
                                 m_imageSize, CV_32FC1, m_leftMap1, m_leftMap2);
    cv::initUndistortRectifyMap(m_cameraMatrixRight, m_distCoeffsRight,
                                 m_rectificationTransformRight, m_projectionMatrixRight,
                                 m_imageSize, CV_32FC1, m_rightMap1, m_rightMap2);

    if (!saveCalibrationResults()) {
        LOG_WARN("保存标定结果失败");
    }
    generateCalibrationReport();
    validateCalibrationResults();

    LOG_INFO("立体标定完成，RMS = {:.6f}", m_rmsError);
    return true;
}

bool StereoCalibrator::saveCalibrationResults() {
    std::string outFile = m_outputDir + "/stereo_calibration.yml";
    try {
        cv::FileStorage fs(outFile, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            LOG_ERROR("无法打开文件写入: {}", outFile);
            return false;
        }

        auto safeWrite = [&](const std::string& name, const cv::Mat& mat) {
            if (!mat.empty()) {
                fs << name << mat;
            } else {
                LOG_WARN("矩阵 {} 为空，跳过写入", name);
            }
        };

        std::string dateTime = std::string(__DATE__) + " " + __TIME__;
        fs << "calibration_date" << dateTime;

        fs << "image_width" << m_imageSize.width;
        fs << "image_height" << m_imageSize.height;
        fs << "board_width" << m_boardSize.width;
        fs << "board_height" << m_boardSize.height;
        fs << "square_size" << m_squareSize;
        fs << "images_used" << m_imagesUsed;
        fs << "rms_error" << m_rmsError;

        // 写入基线长度（平移向量的模长）
        fs << "baseline_meters" << cv::norm(m_translationVector);

        safeWrite("camera_matrix_left", m_cameraMatrixLeft);
        safeWrite("distortion_coefficients_left", m_distCoeffsLeft);
        safeWrite("camera_matrix_right", m_cameraMatrixRight);
        safeWrite("distortion_coefficients_right", m_distCoeffsRight);
        safeWrite("rotation_matrix", m_rotationMatrix);
        safeWrite("translation_vector", m_translationVector);
        safeWrite("essential_matrix", m_essentialMatrix);
        safeWrite("fundamental_matrix", m_fundamentalMatrix);
        safeWrite("rectification_transform_left", m_rectificationTransformLeft);
        safeWrite("rectification_transform_right", m_rectificationTransformRight);
        safeWrite("projection_matrix_left", m_projectionMatrixLeft);
        safeWrite("projection_matrix_right", m_projectionMatrixRight);
        safeWrite("disparity_to_depth_mapping_matrix", m_disparityToDepthMappingMatrix);

        fs << "valid_roi_left" << m_validRoiLeft;
        fs << "valid_roi_right" << m_validRoiRight;

        fs.release();
        LOG_INFO("标定结果已保存: {}", outFile);
        return true;
    } catch (const cv::Exception& e) {
        LOG_ERROR("保存标定结果 OpenCV 异常: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        LOG_ERROR("保存标定结果异常: {}", e.what());
        return false;
    }
}

bool StereoCalibrator::generateCalibrationReport() {
    std::string reportFile = m_outputDir + "/calibration_report.txt";
    std::ofstream f(reportFile);
    if (!f.is_open()) {
        LOG_ERROR("无法创建报告文件: {}", reportFile);
        return false;
    }

    auto matToString = [](const cv::Mat& mat) -> std::string {
        if (mat.empty()) return "空矩阵";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        for (int r = 0; r < mat.rows; ++r) {
            oss << "      [";
            for (int c = 0; c < mat.cols; ++c) {
                if (mat.type() == CV_64F)
                    oss << std::setw(12) << mat.at<double>(r, c);
                else if (mat.type() == CV_32F)
                    oss << std::setw(12) << mat.at<float>(r, c);
                else
                    oss << std::setw(12) << mat.at<int>(r, c);
                if (c < mat.cols - 1) oss << ", ";
            }
            oss << "]\n";
        }
        return oss.str();
    };

    f << "OrangePiZero3-StereoDepth 立体标定详细报告\n";
    f << "============================================\n\n";

    f << "标定时间: " << __DATE__ << " " << __TIME__ << "\n\n";

    f << "【标定参数】\n";
    f << "  图像尺寸: " << m_imageSize.width << " x " << m_imageSize.height << "\n";
    f << "  棋盘格内角点: " << m_boardSize.width << " x " << m_boardSize.height << "\n";
    f << "  方格物理尺寸: " << std::fixed << std::setprecision(3) << m_squareSize << " 米\n";
    f << "  有效图像对数量: " << m_imagesUsed << "\n\n";

    f << "【标定结果】\n";
    f << "  RMS 重投影误差: " << std::setprecision(6) << m_rmsError << "\n";
    f << "  基线长度: " << cv::norm(m_translationVector) << " 米\n\n";

    f << "【左相机内参矩阵】\n";
    f << matToString(m_cameraMatrixLeft) << "\n";
    f << "【左相机畸变系数 (k1, k2, p1, p2, k3)】\n";
    f << matToString(m_distCoeffsLeft) << "\n";

    f << "【右相机内参矩阵】\n";
    f << matToString(m_cameraMatrixRight) << "\n";
    f << "【右相机畸变系数】\n";
    f << matToString(m_distCoeffsRight) << "\n";

    f << "【旋转矩阵 R】\n";
    f << matToString(m_rotationMatrix) << "\n";
    f << "【平移向量 T (米)】\n";
    f << matToString(m_translationVector) << "\n";

    f << "【本质矩阵 E】\n";
    f << matToString(m_essentialMatrix) << "\n";
    f << "【基础矩阵 F】\n";
    f << matToString(m_fundamentalMatrix) << "\n";

    f << "【左有效区域 ROI】 (x, y, width, height): "
      << m_validRoiLeft.x << ", " << m_validRoiLeft.y << ", "
      << m_validRoiLeft.width << ", " << m_validRoiLeft.height << "\n";
    f << "【右有效区域 ROI】: "
      << m_validRoiRight.x << ", " << m_validRoiRight.y << ", "
      << m_validRoiRight.width << ", " << m_validRoiRight.height << "\n\n";

    f << "【校正映射文件】\n";
    f << "  左校正映射已计算并保存在内存中，可通过 StereoPipeline 加载。\n";
    f << "  右校正映射已计算并保存在内存中。\n\n";

    f << "【验证图像】\n";
    f << "  灰度校正验证图: " << m_outputDir << "/rectification_validation_gray.jpg\n";
    f << "  彩色校正验证图: " << m_outputDir << "/rectification_validation_color.jpg\n";
    f << "  图中蓝色水平线用于检查极线对齐，绿色矩形为有效区域 ROI。\n";

    f.close();
    LOG_INFO("标定报告已保存: {}", reportFile);
    return true;
}

bool StereoCalibrator::validateCalibrationResults() {
    auto pairs = findCalibrationImagePairs();
    if (pairs.empty()) {
        LOG_WARN("没有图像可用于验证");
        return false;
    }

    const auto& [leftPath, rightPath] = pairs[0];
    cv::Mat leftColor = cv::imread(leftPath, cv::IMREAD_COLOR);
    cv::Mat rightColor = cv::imread(rightPath, cv::IMREAD_COLOR);
    if (leftColor.empty() || rightColor.empty()) {
        LOG_WARN("无法读取验证图像: {} / {}", leftPath, rightPath);
        return false;
    }

    cv::Mat leftGray, rightGray;
    cv::cvtColor(leftColor, leftGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightColor, rightGray, cv::COLOR_BGR2GRAY);

    if (leftGray.size() != m_imageSize) {
        cv::resize(leftGray, leftGray, m_imageSize);
        cv::resize(leftColor, leftColor, m_imageSize);
    }
    if (rightGray.size() != m_imageSize) {
        cv::resize(rightGray, rightGray, m_imageSize);
        cv::resize(rightColor, rightColor, m_imageSize);
    }

    cv::Mat leftRectGray, rightRectGray;
    cv::remap(leftGray, leftRectGray, m_leftMap1, m_leftMap2, cv::INTER_LINEAR);
    cv::remap(rightGray, rightRectGray, m_rightMap1, m_rightMap2, cv::INTER_LINEAR);

    cv::Mat leftRectColor, rightRectColor;
    cv::remap(leftColor, leftRectColor, m_leftMap1, m_leftMap2, cv::INTER_LINEAR);
    cv::remap(rightColor, rightRectColor, m_rightMap1, m_rightMap2, cv::INTER_LINEAR);

    cv::Mat grayPair, colorPair;
    cv::hconcat(leftRectGray, rightRectGray, grayPair);
    cv::hconcat(leftRectColor, rightRectColor, colorPair);

    int lineSpacing = 50;
    for (int y = 0; y < grayPair.rows; y += lineSpacing) {
        cv::line(grayPair, cv::Point(0, y), cv::Point(grayPair.cols - 1, y),
                 cv::Scalar(255, 255, 255), 1);
        cv::line(colorPair, cv::Point(0, y), cv::Point(colorPair.cols - 1, y),
                 cv::Scalar(255, 0, 0), 1);
    }

    cv::Rect leftRoi = m_validRoiLeft;
    cv::Rect rightRoi = m_validRoiRight;
    rightRoi.x += leftRectGray.cols;

    cv::rectangle(grayPair, leftRoi, cv::Scalar(255, 255, 255), 2);
    cv::rectangle(grayPair, rightRoi, cv::Scalar(255, 255, 255), 2);
    cv::rectangle(colorPair, leftRoi, cv::Scalar(0, 255, 0), 2);
    cv::rectangle(colorPair, rightRoi, cv::Scalar(0, 255, 0), 2);

    std::string grayOut = m_outputDir + "/rectification_validation_gray.jpg";
    std::string colorOut = m_outputDir + "/rectification_validation_color.jpg";
    cv::imwrite(grayOut, grayPair);
    cv::imwrite(colorOut, colorPair);

    LOG_INFO("验证图像已保存: {} 和 {}", grayOut, colorOut);
    return true;
}

} // namespace calibration
} // namespace stereo_depth
