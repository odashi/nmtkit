# nmtkit frontend executables

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -Werror")
endif()

# Boost configurations
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
)
MESSAGE("-- Boost include directory: " ${Boost_INCLUDE_DIRS})

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/submodules/dynet
  ${CMAKE_SOURCE_DIR}/submodules/mteval
  ${CMAKE_SOURCE_DIR}/submodules/spdlog/include
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(train train.cc)
add_executable(decode decode.cc)

target_link_libraries(train ${Boost_LIBRARIES} nmtkit mteval)
target_link_libraries(decode ${Boost_LIBRARIES} nmtkit)
