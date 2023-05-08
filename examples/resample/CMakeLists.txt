set(TARGET_NAME Resample)

file(GLOB SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
file(GLOB CU_SRCS "${CMAKE_CURRENT_LIST_DIR}/*.cu")

add_executable(${TARGET_NAME}
	${SRCS}
    ${CU_SRCS}
    "${DEPENDENCIES_DIR}/glad/src/glad.c"
    ${IMGUI_SRCS}
    ${IMPLOT_SRCS})
target_include_directories(${TARGET_NAME}
    PRIVATE
	${DEPENDENCIES_DIR}
    "${DEPENDENCIES_DIR}/glad/include"
    ${GLFW_INCLUDE_DIR}
    ${IMGUI_INC_DIR}
    ${IMPLOT_INC_DIR})
target_link_libraries(${TARGET_NAME}
    PRIVATE
    glfw
    dpbxvdb
)
target_compile_options(
    ${TARGET_NAME}
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
