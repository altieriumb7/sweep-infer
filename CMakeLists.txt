cmake_minimum_required(VERSION 3.18)
project(sweep VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Compiler flags for AVX-2 and optimization
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -mavx2 -mfma -fPIC)
endif()

# Public Headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/synaptic_fold/kernels
)

# Core Sources
set(SWEEP_SOURCES
    src/sweep.c
    synaptic_fold/kernels/synaptic_entropy_topk_avx2.c
    synaptic_fold/kernels/synaptic_ngram_router_avx2.c
    synaptic_fold/kernels/synaptic_tree_avx2.c
    synaptic_fold/kernels/synaptic_column_avx2.c
)

# Shared Library: libsweep.so / sweep.dll
add_library(sweep SHARED ${SWEEP_SOURCES})
set_target_properties(sweep PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 2
    OUTPUT_NAME "sweep"
)

target_compile_definitions(sweep PRIVATE SWEEP_BUILD_DLL)

# Installation targets
install(TARGETS sweep DESTINATION lib)
install(FILES include/sweep.h DESTINATION include)

message(STATUS "SWEEP v${PROJECT_VERSION} configured successfully.")
