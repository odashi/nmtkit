# nmtkit unit tests

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -Wall -Werror -g")
endif()

# Boost configurations
if(DEFINED BOOST_ROOT OR DEFINED BOOSTROOT OR DEFINED ENV{BOOST_ROOT} OR DEFINED ENV{BOOSTROOT})
  set(Boost_NO_BOOST_CMAKE ON)
  set(Boost_NO_SYSTEM_PATHS ON)
  set(Boost_REALPATH ON)
endif()
find_package(Boost 1.50.0 REQUIRED COMPONENTS
  unit_test_framework
)
MESSAGE("-- Boost include directory: " ${Boost_INCLUDE_DIRS})

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/submodules/dynet
  ${EIGEN3_INCLUDE_DIR}
)

# test definitions
function(nmtkit_test name)
  add_executable(${name}_test ${name}_test.cc)
  target_link_libraries(${name}_test nmtkit ${Boost_LIBRARIES})
  add_test(
    NAME ${name}_test
    COMMAND ${name}_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

nmtkit_test(array)
nmtkit_test(batch_converter)
nmtkit_test(bpe_vocabulary)
nmtkit_test(character_vocabulary)
nmtkit_test(corpus)
nmtkit_test(inference_graph)
nmtkit_test(monotone_sampler)
nmtkit_test(random)
nmtkit_test(sorted_random_sampler)
nmtkit_test(unicode)
nmtkit_test(word_vocabulary)

