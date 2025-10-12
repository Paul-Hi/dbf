message(STATUS "==================================================")
message(STATUS "===================== PyDBF ======================")
message(STATUS "==================================================")

pybind11_add_module(pydbf SHARED pydbf_bindings.cpp)

set_target_properties(pydbf
    PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

if(WIN32)
    set_target_properties(pydbf PROPERTIES SUFFIX ".pyd")
endif()

# Try to find torch_python library from possible locations
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS
        ${torch_SOURCE_DIR}/lib
    NO_DEFAULT_PATH
)

target_link_libraries(pydbf PRIVATE dbf pybind11::module "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
target_include_directories(pydbf PRIVATE ${DBF_INCLUDE_DIR})

set_target_properties(pydbf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

if(WIN32)
    set_target_properties(pydbf PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
    )
endif()

install(TARGETS pydbf
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/pydbf
    ARCHIVE DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/pydbf
    RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/pydbf
)
