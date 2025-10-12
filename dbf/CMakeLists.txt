message(STATUS "==================================================")
message(STATUS "====================== DBF =======================")
message(STATUS "==================================================")

add_library(dbf INTERFACE)
target_include_directories(dbf INTERFACE ${DBF_INCLUDE_DIR})

set_target_properties(dbf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dbf PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_target_properties(dbf PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(dbf INTERFACE DEBUG=1)
else()
    target_compile_definitions(dbf INTERFACE DEBUG=0)
endif()

target_link_libraries(dbf INTERFACE "${TORCH_LIBRARIES}")
