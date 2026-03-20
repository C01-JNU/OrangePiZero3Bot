cmake_minimum_required(VERSION 3.16)
project(OrangePiZero3Bot VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 从 config/cmake.yaml 读取编译选项
find_package(yaml-cpp REQUIRED)
set(CMAKE_CONFIG_FILE ${CMAKE_CURRENT_SOURCE_DIR}/config/cmake.yaml)
if(EXISTS ${CMAKE_CONFIG_FILE})
    file(READ ${CMAKE_CONFIG_FILE} CMAKE_YAML_CONTENT)
    string(REPLACE "\n" ";" CMAKE_YAML_LINES "${CMAKE_YAML_CONTENT}")
    foreach(line ${CMAKE_YAML_LINES})
        if(line MATCHES "use_gpu:[ \t]*([^ \t\n]+)")
            set(USE_GPU_FROM_YAML ${CMAKE_MATCH_1})
            string(TOLOWER "${USE_GPU_FROM_YAML}" USE_GPU_FROM_YAML)
            if(USE_GPU_FROM_YAML STREQUAL "true" OR USE_GPU_FROM_YAML STREQUAL "1")
                set(WITH_VULKAN ON)
            else()
                set(WITH_VULKAN OFF)
            endif()
        endif()
    endforeach()
else()
    message(WARNING "config/cmake.yaml not found, using default WITH_VULKAN=OFF")
    set(WITH_VULKAN OFF)
endif()

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
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
add_subdirectory(src/census)          # census 模块

# 主程序
add_executable(stereo_depth src/main.cpp)
target_link_libraries(stereo_depth PRIVATE
    camera
    calibration_utils
    census
    utils
    network
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
)
set_target_properties(stereo_depth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 批量测试程序
add_executable(test_stereo_depth src/test.cpp)
target_link_libraries(test_stereo_depth PRIVATE
    camera
    calibration_utils
    census
    utils
    network
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
)
set_target_properties(test_stereo_depth PROPERTIES
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
    yaml-cpp
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
    yaml-cpp
    Threads::Threads
)
set_target_properties(test_calibration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 定义辅助函数来复制公共数据
function(copy_common_data TARGET_NAME)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/config
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/config $<TARGET_FILE_DIR:${TARGET_NAME}>/config/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/calibration_results
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/calibration_results $<TARGET_FILE_DIR:${TARGET_NAME}>/calibration_results/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/images/test
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/test $<TARGET_FILE_DIR:${TARGET_NAME}>/images/test
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/images/output
        COMMENT "Copying common data to ${TARGET_NAME} directory"
    )
endfunction()

copy_common_data(stereo_depth)
copy_common_data(test_stereo_depth)

# 为 stereo_calibrator 额外复制 images/calibration
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
    COMMENT "Copying data to stereo_calibrator directory"
)

# 为 test_calibration 额外复制 images/calibration
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
    COMMENT "Copying data to test_calibration directory"
)

install(TARGETS stereo_depth DESTINATION bin)

message(STATUS "=========================================")
message(STATUS "OrangePiZero3Bot Configuration")
message(STATUS "=========================================")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "WITH_VULKAN (from config/cmake.yaml): ${WITH_VULKAN}")
message(STATUS "Output binaries: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=========================================")
