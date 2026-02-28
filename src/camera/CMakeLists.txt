add_library(camera
    camera_factory.cpp
    chusei_camera.cpp
    mock_camera.cpp
)
target_include_directories(camera PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(camera
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
)
set_target_properties(camera PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
