message(STATUS "==================================================")
message(STATUS "====================== DBF =======================")
message(STATUS "==================================================")

file(GLOB SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
add_library(dbf STATIC ${SRC})
target_include_directories(dbf PUBLIC ${DBF_INCLUDE_DIR})
target_link_libraries(dbf "${TORCH_LIBRARIES}")

set_target_properties(dbf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(dbf PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_target_properties(dbf PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(dbf PUBLIC DEBUG=1)
else()
    target_compile_definitions(dbf PUBLIC DEBUG=0)
endif()
