cmake_minimum_required(VERSION 3.29) # Adjust the version as necessary
cmake_policy(SET CMP0048 NEW) # Allow project() to set VERSION

set(FAISS_LANGUAGES CXX)
if(FAISS_ENABLE_GPU)
  list(APPEND FAISS_LANGUAGES CUDA)
endif()

if(FAISS_ENABLE_RAFT)
  include(cmake/thirdparty/fetch_rapids.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)
  rapids_cuda_init_architectures(faiss)
  rapids_cuda_init_architectures(pyfaiss)
  rapids_cuda_init_architectures(faiss_c_library)
endif()

project(faiss VERSION 1.8.0
  DESCRIPTION "A library for efficient similarity search and clustering of dense vectors."
  HOMEPAGE_URL "https://github.com/facebookresearch/faiss"
  LANGUAGES ${FAISS_LANGUAGES}
)

include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Valid values are "generic", "avx2", "avx512".
option(FAISS_OPT_LEVEL "" "generic")
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." ON)
option(FAISS_ENABLE_RAFT "Enable RAFT for GPU indexes." OFF)
option(FAISS_ENABLE_PYTHON "Build Python extension." ON)
option(FAISS_ENABLE_C_API "Build C API." OFF)

if(FAISS_ENABLE_GPU)
  find_package(CUDA REQUIRED)
  enable_language(CUDA)
endif()

if(FAISS_ENABLE_RAFT AND NOT TARGET raft::raft)
  find_package(raft COMPONENTS compiled distributed)
endif()

add_subdirectory(faiss)

if(FAISS_ENABLE_GPU)
  add_subdirectory(faiss/gpu)
endif()

if(FAISS_ENABLE_PYTHON)
  add_subdirectory(faiss/python)
endif()

if(FAISS_ENABLE_C_API)
  add_subdirectory(c_api)
endif()

add_subdirectory(demos)
add_subdirectory(benchs)
add_subdirectory(tutorial/cpp)

# CTest must be included in the top level to enable `make test` target.
include(CTest)

if(BUILD_TESTING)
  add_subdirectory(tests)
  if(FAISS_ENABLE_GPU)
    add_subdirectory(faiss/gpu/test)
  endif()
endif()