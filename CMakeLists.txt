cmake_minimum_required(VERSION 3.16)
project(OrangePiZero3Bot VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------
# 1. 使用 ExternalProject 编译 MNN（只编译推理库 + Vulkan 后端）
# ------------------------------------------------------------------
include(ExternalProject)

set(MNN_PREFIX ${CMAKE_BINARY_DIR}/thirdparty/mnn)      # 安装目录
set(MNN_SOURCE_DIR ${CMAKE_SOURCE_DIR}/thirdparty/MNN)  # 源码位置

if(NOT EXISTS ${MNN_SOURCE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "MNN source not found at ${MNN_SOURCE_DIR}. Please run: git submodule update --init --recursive")
endif()

ExternalProject_Add(MNN_project
    SOURCE_DIR ${MNN_SOURCE_DIR}
    BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/mnn-build
    INSTALL_DIR ${MNN_PREFIX}
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=${MNN_PREFIX}
        -DMNN_BUILD_SHARED_LIBS=ON
        -DMNN_VULKAN=ON
        -DMNN_SEP_BUILD=ON
        -DMNN_BUILD_MINI=ON
        -DMNN_BUILD_CONVERTER=ON
        -DMNN_OPENCL=OFF
        -DMNN_CUDA=OFF
        -DMNN_METAL=OFF
        -DMNN_ARM82=ON
        -DMNN_USE_THREAD_POOL=ON
    BUILD_COMMAND $(MAKE) MNN
    INSTALL_COMMAND $(MAKE) install
)

# ------------------------------------------------------------------
# 2. 创建导入库目标，供主程序链接
# ------------------------------------------------------------------
add_library(MNN SHARED IMPORTED GLOBAL)
add_dependencies(MNN MNN_project)

set(MNN_LIBRARY ${MNN_PREFIX}/lib/libMNN.so)
set(MNN_VULKAN_LIB ${MNN_PREFIX}/lib/libMNN_Vulkan.so)

set_target_properties(MNN PROPERTIES
    IMPORTED_LOCATION ${MNN_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${MNN_PREFIX}/include
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 QUIET)

if(WITH_VULKAN)
    find_package(Vulkan REQUIRED)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src/camera)
add_subdirectory(src/calibration)
add_subdirectory(src/utils)
add_subdirectory(src/network)
add_subdirectory(src/preprocess)
add_subdirectory(src/inference)

# ------------------------------------------------------------------
# 3. 可执行程序（主程序、测试、标定等）
# ------------------------------------------------------------------
# 主程序（start）
add_executable(start src/main.cpp)
target_link_libraries(start PRIVATE
    camera
    calibration_utils
    preprocess
    utils
    network
    inference
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Threads::Threads
)
set_target_properties(start PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 测试程序（test）
add_executable(test src/test.cpp)
target_link_libraries(test PRIVATE
    camera
    calibration_utils
    preprocess
    utils
    network
    inference
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Threads::Threads
)
set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 标定程序
add_executable(stereo_calibrator src/calibration/stereo_calibrator_main.cpp)
target_link_libraries(stereo_calibrator PRIVATE
    calibration_utils
    utils
    network
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Threads::Threads
)
set_target_properties(stereo_calibrator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 校正测试程序
add_executable(test_calibration src/calibration/test_calibration.cpp)
target_link_libraries(test_calibration PRIVATE
    calibration_utils
    utils
    network
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Threads::Threads
)
set_target_properties(test_calibration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ------------------------------------------------------------------
# 4. 辅助函数：复制公共数据（config, images, calibration_results）
# ------------------------------------------------------------------
function(copy_common_data TARGET_NAME)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/config
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/config $<TARGET_FILE_DIR:${TARGET_NAME}>/config/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/calibration_results
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/calibration_results $<TARGET_FILE_DIR:${TARGET_NAME}>/calibration_results/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/images/test
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/test $<TARGET_FILE_DIR:${TARGET_NAME}>/images/test
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/images/output
        COMMENT "复制公共数据到 ${TARGET_NAME} 目录"
    )
endfunction()

# 为 start 和 test 复制公共数据
copy_common_data(start)
copy_common_data(test)

# ------------------------------------------------------------------
# 5. 辅助函数：复制模型文件（若存在）
# ------------------------------------------------------------------
function(copy_models TARGET_NAME)
    if(EXISTS ${CMAKE_SOURCE_DIR}/models)
        add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/models
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:${TARGET_NAME}>/models/
            COMMENT "复制模型文件到 ${TARGET_NAME} 目录"
        )
    else()
        message(STATUS "未找到 models 目录，跳过模型复制（请手动将模型文件放入 bin/models 下）")
    endif()
endfunction()

copy_models(start)
copy_models(test)

# ------------------------------------------------------------------
# 6. 为 stereo_calibrator 和 test_calibration 额外复制 calibration 图像
# ------------------------------------------------------------------
add_custom_command(TARGET stereo_calibrator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_calibrator>/config
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/config $<TARGET_FILE_DIR:stereo_calibrator>/config/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_calibrator>/calibration_results
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/calibration_results $<TARGET_FILE_DIR:stereo_calibrator>/calibration_results/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_calibrator>/images/test
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/test $<TARGET_FILE_DIR:stereo_calibrator>/images/test
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_calibrator>/images/calibration
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/calibration $<TARGET_FILE_DIR:stereo_calibrator>/images/calibration
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:stereo_calibrator>/images/output
    COMMENT "复制标定数据到 stereo_calibrator 目录"
)

add_custom_command(TARGET test_calibration POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_calibration>/config
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/config $<TARGET_FILE_DIR:test_calibration>/config/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_calibration>/calibration_results
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/calibration_results $<TARGET_FILE_DIR:test_calibration>/calibration_results/
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_calibration>/images/test
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/test $<TARGET_FILE_DIR:test_calibration>/images/test
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_calibration>/images/calibration
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/calibration $<TARGET_FILE_DIR:test_calibration>/images/calibration
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:test_calibration>/images/output
    COMMENT "复制标定数据到 test_calibration 目录"
)

install(TARGETS start DESTINATION bin)

message(STATUS "=========================================")
message(STATUS "FishTotem Configuration")
message(STATUS "=========================================")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output binaries: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=========================================")
