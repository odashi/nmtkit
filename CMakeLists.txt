cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(nmtkit VERSION 0.1.2 LANGUAGES CXX)

# utility to check if required option is set.
function(my_check_option name)
  if(DEFINED ENV{${name}} AND NOT DEFINED ${name})
    set(${name} "$ENV{${name}}")
  endif()
  if(${name})
    MESSAGE("-- ${name}: ${${name}}")
  else()
    MESSAGE(FATAL_ERROR "Please give the variable `${name}`.")
  endif()
endfunction()

# compiler requirements and settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror")
endif()

# Boost
if(DEFINED BOOST_ROOT OR DEFINED BOOSTROOT OR DEFINED ENV{BOOST_ROOT} OR DEFINED ENV{BOOSTROOT})
  set(Boost_NO_BOOST_CMAKE ON)
  set(Boost_NO_SYSTEM_PATHS ON)
  set(Boost_REALPATH ON)
endif()
find_package(Boost 1.50.0 REQUIRED COMPONENTS
  filesystem
  program_options
  serialization
  system
  unit_test_framework
)

# Eigen3/DyNet/MTEval
my_check_option(EIGEN3_INCLUDE_DIR)
my_check_option(DYNET_INCLUDE_DIR)
my_check_option(DYNET_LIBRARY_DIR)
my_check_option(MTEVAL_INCLUDE_DIR)
my_check_option(MTEVAL_LIBRARY_DIR)

# options
option(USE_GPU
  "If ON, NMTKit would use the CUDA backend."
  OFF
)
option(USE_VANILLA_LSTM
  "If ON, NMTKit would use VanillaLSTMBuilder module instead of LSTMBuilder."
  OFF
)

# CUDA
if(USE_GPU)
  my_check_option(CUDA_ROOT)
  set(CUDA_INCLUDE_DIR "${CUDA_ROOT}/include")
  set(CUDA_LIBRARY_DIR "${CUDA_ROOT}/lib64")
  set(DYNET_LIBRARIES "gdynet")
  MESSAGE("-- Build NMTKit with gDyNet (CUDA).")
else()
  set(DYNET_LIBRARIES "dynet")
  MESSAGE("-- Build NMTKit with DyNet (CPU).")
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/config.h
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_BINARY_DIR}
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${DYNET_INCLUDE_DIR}
  ${MTEVAL_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/submodules/spdlog/include
)
if(USE_GPU)
  include_directories(${CUDA_INCLUDE_DIR})
endif()

link_directories(
  ${Boost_LIBRARY_DIRS}
  ${DYNET_LIBRARY_DIR}
  ${MTEVAL_LIBRARY_DIR}
)
if(USE_GPU)
  link_directories(${CUDA_LIBRARY_DIR})
endif()

set(BUILD_SHARED_LIBS ON)

enable_testing()

# nmtkit sources
add_subdirectory(bin)
add_subdirectory(nmtkit)
add_subdirectory(test)
