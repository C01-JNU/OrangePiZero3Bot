add_library(inference
    yolo_detector.cpp
)

target_include_directories(inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(inference PRIVATE
    MNN
    utils
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Threads::Threads
)

set_target_properties(inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# 测试可执行程序
add_executable(test_inference test_inference_main.cpp)
target_link_libraries(test_inference PRIVATE
    inference
    utils
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Threads::Threads
)
set_target_properties(test_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
