set(GGML_HEADERS_BITNET ../include/ggml-bitnet.h)
set(GGML_SOURCES_BITNET ggml-bitnet-mad.cpp ggml-bitnet-lut.cpp)

add_library(bitnet STATIC ${GGML_SOURCES_BITNET})
target_include_directories(bitnet
    PUBLIC
        ../include
        ${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
        ${CMAKE_SOURCE_DIR}/3rdparty/llama.cpp/ggml/src
)

set_target_properties(bitnet PROPERTIES PUBLIC_HEADER ../include/ggml-bitnet.h)

install(TARGETS bitnet
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)

install(DIRECTORY ../include/
    DESTINATION include
)

if (NOT (CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID STREQUAL "GNU") OR
    NOT (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(FATAL_ERROR "Clang or GCC is required for Bitnet.cpp compilation")
endif()
