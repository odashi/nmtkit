lib_LTLIBRARIES = libnmtkit.la

AM_CXXFLAGS = \
  -I$(top_builddir)/new_src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS) \
  $(CUDA_CPPFLAGS)

libnmtkit_la_SOURCES = \
  init.cc \
  random.cc \
  unicode.cc

