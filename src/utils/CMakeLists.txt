add_library(utils
    config.cpp
    logger.cpp
)
target_include_directories(utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(utils
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
)
set_target_properties(utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
