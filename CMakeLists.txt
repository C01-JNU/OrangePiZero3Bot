cmake_minimum_required(VERSION 3.16)
project(OrangePiZero3Bot VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 QUIET)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src/camera)
add_subdirectory(src/calibration)
add_subdirectory(src/cpu_stereo)
add_subdirectory(src/utils)

# 主程序
add_executable(stereo_depth src/main.cpp)
target_link_libraries(stereo_depth PRIVATE
    camera
    calibration_utils
    cpu_stereo
    utils
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
    cpu_stereo
    utils
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
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
    Threads::Threads
)
set_target_properties(test_calibration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 数据复制（对所有可执行文件）
function(copy_data_to_target TARGET_NAME)
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/config
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/config $<TARGET_FILE_DIR:${TARGET_NAME}>/config/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/calibration_results
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/calibration_results $<TARGET_FILE_DIR:${TARGET_NAME}>/calibration_results/
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/images/test
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/test $<TARGET_FILE_DIR:${TARGET_NAME}>/images/test
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/images/calibration
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/images/calibration $<TARGET_FILE_DIR:${TARGET_NAME}>/images/calibration
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${TARGET_NAME}>/images/output
        COMMENT "Copying data to ${TARGET_NAME} directory"
    )
endfunction()

copy_data_to_target(stereo_depth)
copy_data_to_target(test_stereo_depth)
copy_data_to_target(stereo_calibrator)
copy_data_to_target(test_calibration)

install(TARGETS stereo_depth DESTINATION bin)

message(STATUS "=========================================")
message(STATUS "OrangePiZero3Bot Configuration")
message(STATUS "=========================================")
message(STATUS "Project version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output binaries: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=========================================")
