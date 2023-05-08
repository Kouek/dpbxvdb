cmake_minimum_required(VERSION 3.18)

project(dpbxvdb LANGUAGES CXX CUDA)

option(DPBXVDB_BUILD_EXAMPLES "Build examples using Depth-box VDB." ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(DEPENDENCIES_DIR "${CMAKE_CURRENT_LIST_DIR}/deps")

add_subdirectory("${DEPENDENCIES_DIR}/glm")

set(TARGET_NAME dpbxvdb)
file(GLOB CU_SRCS "${CMAKE_CURRENT_LIST_DIR}/dpbxvdb/kernels/*.cu")
add_library(${TARGET_NAME} ${CU_SRCS})
target_include_directories(${TARGET_NAME}
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${TARGET_NAME} PUBLIC glm::glm)
target_compile_options(
    ${TARGET_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

if(DPBXVDB_BUILD_EXAMPLES)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
	add_subdirectory("${DEPENDENCIES_DIR}/glfw")

	set(IMGUI_INC_DIR "${DEPENDENCIES_DIR}/imgui")
	file(GLOB IMGUI_SRCS "${DEPENDENCIES_DIR}/imgui/*.cpp")

	set(IMPLOT_INC_DIR "${DEPENDENCIES_DIR}/implot")
	file(GLOB IMPLOT_SRCS "${DEPENDENCIES_DIR}/implot/*.cpp")

	set(EXAMPLES_DIR "${CMAKE_CURRENT_LIST_DIR}/examples")
	add_subdirectory("${EXAMPLES_DIR}/resample")
endif()
