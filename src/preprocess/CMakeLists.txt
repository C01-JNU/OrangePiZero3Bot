# Preprocess 模块：去噪 + Census 变换

set(PREPROCESS_SOURCES
    preprocess.cpp
    denoiser.cpp
    census.cpp
)

add_library(preprocess ${PREPROCESS_SOURCES})

target_include_directories(preprocess PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(preprocess
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    utils
)

set_target_properties(preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
