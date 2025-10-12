message(STATUS "==================================================")
message(STATUS "================= DBF Sandbox ====================")
message(STATUS "==================================================")

add_executable(dbf_sandbox main.cpp)
target_include_directories(dbf_sandbox PRIVATE ${DBF_INCLUDE_DIR})
target_link_libraries(dbf_sandbox PRIVATE dbf)

set_target_properties(dbf_sandbox PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET dbf_sandbox PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_definitions(dbf_sandbox PRIVATE DEBUG=1)
else()
  target_compile_definitions(dbf_sandbox PRIVATE DEBUG=0)
endif()

target_link_libraries(dbf_sandbox PRIVATE "${TORCH_LIBRARIES}")
