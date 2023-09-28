cmake_minimum_required(VERSION 3.0.0)
project(vmd VERSION 0.1.0 LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50 52 60 61 62 75)
endif()

enable_language(CUDA)

message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# no testing 

set( project_sources
   src/main.cpp
   src/AudioProc.cu
   src/SpatialProcParent.cu
   src/XCorrProcParent.cu
   src/FileHandler.cpp
   src/SQLUtils.cpp
)

add_executable(${PROJECT_NAME} ${project_sources})

target_include_directories(${PROJECT_NAME}
   PRIVATE
   /usr/include/cppconn/)

target_link_libraries(vmd mysqlcppconn)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
