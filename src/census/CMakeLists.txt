# Census 变换模块

set(CENSUS_SOURCES
    census_transform.cpp
    census_transform_cpu.cpp
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

    # 着色器编译
    set(SHADER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
    set(SHADER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    file(MAKE_DIRECTORY ${SHADER_BINARY_DIR})

    file(GLOB SHADER_SOURCES ${SHADER_SOURCE_DIR}/*.comp)
    set(SPV_TARGETS)
    foreach(shader ${SHADER_SOURCES})
        get_filename_component(name ${shader} NAME_WLE)
        set(spv_file ${SHADER_BINARY_DIR}/${name}.comp.spv)
        add_custom_command(
            OUTPUT ${spv_file}
            COMMAND ${SHADER_COMPILER} ${SHADER_COMPILER_ARGS} ${spv_file} ${shader}
            DEPENDS ${shader}
            COMMENT "Compiling shader ${name}"
        )
        list(APPEND SPV_TARGETS ${spv_file})
    endforeach()

    add_custom_target(generate_census_shaders ALL DEPENDS ${SPV_TARGETS})

    foreach(spv ${SPV_TARGETS})
        get_filename_component(spv_name ${spv} NAME)
        add_custom_command(TARGET generate_census_shaders POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin/shaders/
            COMMAND ${CMAKE_COMMAND} -E copy ${spv} ${CMAKE_BINARY_DIR}/bin/shaders/${spv_name}
        )
    endforeach()
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
