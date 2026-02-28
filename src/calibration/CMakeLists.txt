add_library(calibration_utils
    calibration_loader.cpp
    stereo_rectifier.cpp
    stereo_calibrator.cpp
)
target_include_directories(calibration_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(calibration_utils
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    yaml-cpp
)
set_target_properties(calibration_utils PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
