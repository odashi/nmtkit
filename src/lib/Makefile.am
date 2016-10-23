lib_LTLIBRARIES = libnmtkit.la

AM_CXXFLAGS = \
  -I$(top_builddir)/src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS)

libnmtkit_la_SOURCES = \
  corpus.cc \
  batch_converter.cc \
  encoder_decoder.cc \
  monotone_sampler.cc \
  random.cc \
  random_sampler.cc \
  sorted_random_sampler.cc \
  vocabulary.cc

