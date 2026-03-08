add_library(network
    udp_streamer.cpp
)
target_include_directories(network PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(network
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    utils
)
set_target_properties(network PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
