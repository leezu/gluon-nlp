cmake_minimum_required(VERSION 3.12)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

project(sentencepiece_nogil LANGUAGES C CXX)

set(PYBIND11_CPP_STANDARD -std=c++17)
add_subdirectory(third_party/pybind11)

SET(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "Enable TCMalloc if available.")
add_subdirectory(third_party/sentencepiece EXCLUDE_FROM_ALL)
target_include_directories(sentencepiece PRIVATE third_party/sentencepiece/)

pybind11_add_module(sentencepiecewrap sentencepiecewrap.cxx)
target_link_libraries(sentencepiecewrap PUBLIC sentencepiece)
target_include_directories(sentencepiecewrap PUBLIC third_party/range-v3/include)
