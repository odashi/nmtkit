lib_LTLIBRARIES = libnmtkit.la

AM_CXXFLAGS = \
  -I$(top_builddir)/new_src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS) \
  $(CUDA_CPPFLAGS)

libnmtkit_la_SOURCES = \
  basic_types.cc \
  character_vocabulary.cc \
  init.cc \
  random.cc \
  simple_text_reader.cc \
  unicode.cc \
  vocabulary.cc \
  word_vocabulary.cc

