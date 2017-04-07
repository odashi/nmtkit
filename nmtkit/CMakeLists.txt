# nmtkit library

set(nmtkit_LIBRARY_HDRS
  array.h
  attention.h
  backward_encoder.h
  bahdanau_decoder.h
  basic_types.h
  batch_converter.h
  bidirectional_encoder.h
  bilinear_attention.h
  bpe_vocabulary.h
  character_vocabulary.h
  corpus.h
  decoder.h
  default_decoder.h
  encoder.h
  encoder_decoder.h
  exception.h
  factories.h
  formatter.h
  forward_encoder.h
  html_formatter.h
  inference_graph.h
  init.h
  luong_decoder.h
  mlp_attention.h
  monotone_sampler.h
  multilayer_perceptron.h
  predictor.h
  random.h
  sampler.h
  serialization_utils.h
  single_text_formatter.h
  softmax_predictor.h
  sorted_random_sampler.h
  unicode.h
  vocabulary.h
  word_vocabulary.h
)

set(nmtkit_LIBRARY_SRCS
  attention.cc
  backward_encoder.cc
  bahdanau_decoder.cc
  batch_converter.cc
  bidirectional_encoder.cc
  bilinear_attention.cc
  bpe_vocabulary.cc
  character_vocabulary.cc
  corpus.cc
  decoder.cc
  default_decoder.cc
  encoder.cc
  encoder_decoder.cc
  factories.cc
  forward_encoder.cc
  html_formatter.cc
  inference_graph.cc
  init.cc
  luong_decoder.cc
  mlp_attention.cc
  monotone_sampler.cc
  multilayer_perceptron.cc
  predictor.cc
  random.cc
  single_text_formatter.cc
  softmax_predictor.cc
  sorted_random_sampler.cc
  unicode.cc
  vocabulary.cc
  word_vocabulary.cc
)

set(BUILD_SHARED_LIBS ON)

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
  serialization
)
MESSAGE("-- Boost include directory: " ${Boost_INCLUDE_DIRS})

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/submodules/dynet
  ${EIGEN3_INCLUDE_DIR}
)

add_library(nmtkit ${nmtkit_LIBRARY_HDRS} ${nmtkit_LIBRARY_SRCS})
if(BACKEND MATCHES "^cuda$")
  MESSAGE("-- Preparing NMTKit with gDyNet (CUDA).")
  target_link_libraries(nmtkit ${Boost_LIBRARIES} gdynet)
else()
  MESSAGE("-- Preparing NMTKit with Dynet (CPU).")
  target_link_libraries(nmtkit ${Boost_LIBRARIES} dynet)
endif()
