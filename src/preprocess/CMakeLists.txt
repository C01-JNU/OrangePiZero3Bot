# Preprocess 模块：去噪 + Census 变换

set(PREPROCESS_SOURCES
    preprocess.cpp
    denoiser.cpp
    census.cpp
)

if(WITH_VULKAN)
    list(APPEND PREPROCESS_SOURCES preprocess_gpu.cpp)
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

    # 编译双边滤波着色器
    set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

    set(BILATERAL_SHADER ${SHADER_SOURCE_DIR}/bilateral.comp)
    set(CENSUS_SHADER ${SHADER_SOURCE_DIR}/census_adaptive.comp)
    set(BILATERAL_SPV ${SHADER_BINARY_DIR}/bilateral.comp.spv)
    set(CENSUS_SPV ${SHADER_BINARY_DIR}/census_adaptive.comp.spv)

    add_custom_command(
        OUTPUT ${BILATERAL_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_BINARY_DIR}
        COMMAND ${SHADER_COMPILER} ${SHADER_COMPILER_ARGS} ${BILATERAL_SPV} ${BILATERAL_SHADER}
        DEPENDS ${BILATERAL_SHADER}
        COMMENT "Compiling bilateral shader"
    )
    add_custom_command(
        OUTPUT ${CENSUS_SPV}
        COMMAND ${SHADER_COMPILER} ${SHADER_COMPILER_ARGS} ${CENSUS_SPV} ${CENSUS_SHADER}
        DEPENDS ${CENSUS_SHADER}
        COMMENT "Compiling census shader"
    )
    add_custom_target(generate_preprocess_shaders ALL
        DEPENDS ${BILATERAL_SPV} ${CENSUS_SPV}
    )

    # 复制到可执行文件目录
    add_custom_command(TARGET generate_preprocess_shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/
        COMMAND ${CMAKE_COMMAND} -E copy ${BILATERAL_SPV} ${CMAKE_BINARY_DIR}/bin/shaders/
        COMMAND ${CMAKE_COMMAND} -E copy ${CENSUS_SPV} ${CMAKE_BINARY_DIR}/bin/shaders/
    )
endif()

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

if(WITH_VULKAN)
    target_compile_definitions(preprocess PRIVATE WITH_VULKAN)
    target_link_libraries(preprocess Vulkan::Vulkan)
endif()

set_target_properties(preprocess PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
