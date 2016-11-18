lib_LTLIBRARIES = libnmtkit.la

AM_CXXFLAGS = \
  -I$(top_builddir)/src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS) \
  $(CUDA_CPPFLAGS)

libnmtkit_la_SOURCES = \
  attention.cc \
  corpus.cc \
  backward_encoder.cc \
  bahdanau_decoder.cc \
  batch_converter.cc \
  bidirectional_encoder.cc \
  bilinear_attention.cc \
  character_vocabulary.cc \
  decoder.cc \
  default_decoder.cc \
  encoder.cc \
  encoder_decoder.cc \
  factories.cc \
  forward_encoder.cc \
  html_formatter.cc \
  inference_graph.cc \
  init.cc \
  mlp_attention.cc \
  monotone_sampler.cc \
  multilayer_perceptron.cc \
  predictor.cc \
  random.cc \
  random_sampler.cc \
  single_text_formatter.cc \
  softmax_predictor.cc \
  sorted_random_sampler.cc \
  vocabulary.cc \
  word_vocabulary.cc

