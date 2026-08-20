# Trimmed from MAGIA SDK's onnx_mobilevit_v2 application. The frozen copy executes only
# original ONNX nodes 170 through 179 and is built as an external SDK application.
set(MAPS_APPLICATION full_mesh_mobilevit_slice)
set(KERNELS_DIR ${CMAKE_SOURCE_DIR}/kernels/spatz_fp16)
set(REFERENCE_KERNELS groupnorm conv2dgemm softmax mul reducesum relu add)

set(REFERENCE_TASK_SOURCES "")
set(REFERENCE_KERNEL_SOURCES "")
set(REFERENCE_KERNEL_INCLUDES "")
foreach(K ${REFERENCE_KERNELS})
  list(APPEND REFERENCE_TASK_SOURCES ${KERNELS_DIR}/${K}/spatz_task/${K}_fp16_spatz_task.c)
  list(APPEND REFERENCE_KERNEL_SOURCES ${KERNELS_DIR}/${K}/src/${K}_fp16_spatz.c)
  list(APPEND REFERENCE_KERNEL_INCLUDES ${KERNELS_DIR}/${K}/include)
endforeach()
list(APPEND REFERENCE_TASK_SOURCES
  ${KERNELS_DIR}/mul/spatz_task/mul_bcast_fp16_spatz_task.c
  ${KERNELS_DIR}/add/spatz_task/add_bcast_fp16_spatz_task.c
)

add_spatz_task(
  TEST_NAME ${MAPS_APPLICATION}
  TASK_SOURCES ${REFERENCE_TASK_SOURCES}
  FIRST_TASK_NAME groupnorm_fp16_spatz_task
  INCLUDE_DIRS ${REFERENCE_KERNEL_INCLUDES}
)

configure_file(src/reference_data.S.in ${CMAKE_CURRENT_BINARY_DIR}/reference_data.S @ONLY)
add_cv32_executable_with_spatz(
  TARGET_NAME ${MAPS_APPLICATION}
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
    ${CMAKE_CURRENT_BINARY_DIR}/reference_data.S
    ${REFERENCE_KERNEL_SOURCES}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/shim
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${REFERENCE_KERNEL_INCLUDES}
)
target_compile_options(${MAPS_APPLICATION} PRIVATE -O2)
