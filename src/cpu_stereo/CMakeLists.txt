add_library(cpu_stereo
    cpu_stereo_matcher.cpp
)
target_include_directories(cpu_stereo PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(cpu_stereo
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
)
set_target_properties(cpu_stereo PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
