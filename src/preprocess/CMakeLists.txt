set(PREPROCESS_SOURCES
    preprocess.cpp
    preprocess_gpu.cpp
    denoiser.cpp
    census.cpp
    gpu_census.cpp
)

find_package(Vulkan REQUIRED)

# 查找着色器编译器
find_program(GLSLC_EXECUTABLE NAMES glslc)
find_program(GLSANGVALIDATOR_EXECUTABLE NAMES glslangValidator)

if(GLSLC_EXECUTABLE)
    set(SHADER_COMPILER ${GLSLC_EXECUTABLE})
    set(SHADER_COMPILER_ARGS -o)
elseif(GLSANGVALIDATOR_EXECUTABLE)
    set(SHADER_COMPILER ${GLSANGVALIDATOR_EXECUTABLE})
    set(SHADER_COMPILER_ARGS -V -o)
else()
    message(FATAL_ERROR "Neither glslc nor glslangValidator found.")
endif()

# 编译着色器
set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

set(CENSUS_SHADER ${SHADER_SOURCE_DIR}/census_adaptive.comp)
set(CENSUS_SPV ${SHADER_BINARY_DIR}/census_adaptive.comp.spv)

add_custom_command(
    OUTPUT ${CENSUS_SPV}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
    COMMAND ${SHADER_COMPILER} ${SHADER_COMPILER_ARGS} ${CENSUS_SPV} ${CENSUS_SHADER}
    DEPENDS ${CENSUS_SHADER}
    COMMENT "Compiling census shader"
)
add_custom_target(generate_preprocess_shaders ALL
    DEPENDS ${CENSUS_SPV}
)

add_custom_command(TARGET generate_preprocess_shaders POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy ${CENSUS_SPV} ${CMAKE_BINARY_DIR}/bin/shaders/
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
    Vulkan::Vulkan
)

set_target_properties(preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
