lib_LTLIBRARIES = libnmtkit.la

AM_CXXFLAGS = \
  -I$(top_builddir)/src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS)

libnmtkit_la_SOURCES = \
  attention.cc \
  corpus.cc \
  batch_converter.cc \
  bidirectional_encoder.cc \
  bilinear_attention.cc \
  encoder.cc \
  encoder_decoder.cc \
  html_formatter.cc \
  inference_graph.cc \
  mlp_attention.cc \
  monotone_sampler.cc \
  multilayer_perceptron.cc \
  random.cc \
  random_sampler.cc \
  single_text_formatter.cc \
  sorted_random_sampler.cc \
  vocabulary.cc

