# Census 变换模块

set(CENSUS_SOURCES
    census_transform.cpp
    census_transform_cpu.cpp
    filters.cpp
)

if(WITH_VULKAN)
    list(APPEND CENSUS_SOURCES census_transform_gpu.cpp)
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

    # 编译整合后的着色器
    set(SHADER_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/shaders/census.comp)
    set(SPV_FILE ${CMAKE_CURRENT_BINARY_DIR}/shaders/census.comp.spv)
    add_custom_command(
        OUTPUT ${SPV_FILE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
        COMMAND ${SHADER_COMPILER} ${SHADER_COMPILER_ARGS} ${SPV_FILE} ${SHADER_SOURCE}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling shader census.comp"
    )
    add_custom_target(generate_census_shaders ALL DEPENDS ${SPV_FILE})

    # 复制到可执行文件目录
    add_custom_command(TARGET generate_census_shaders POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/
        COMMAND ${CMAKE_COMMAND} -E copy ${SPV_FILE} ${CMAKE_BINARY_DIR}/bin/shaders/census.comp.spv
    )
endif()

add_library(census ${CENSUS_SOURCES})

target_include_directories(census PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(census
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    utils
)

if(WITH_VULKAN)
    target_compile_definitions(census PRIVATE WITH_VULKAN)
    target_link_libraries(census Vulkan::Vulkan)
endif()

set_target_properties(census PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
