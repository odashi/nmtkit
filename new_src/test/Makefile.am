AM_CXXFLAGS = \
  -I$(top_builddir)/new_src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS) \
  $(CUDA_CPPFLAGS)

LDADD = \
  -lstdc++ \
  $(top_builddir)/new_src/lib/libnmtkit.la \
  $(BOOST_UNIT_TEST_FRAMEWORK_LIB) \
  $(BOOST_SERIALIZATION_LIB) \
  $(DYNET_LDFLAGS) \
  $(CUDA_LDFLAGS)

check_PROGRAMS = \
  array_test \
  random_test \
  unicode_test

TESTS = \
  array_test \
  random_test \
  unicode_test

array_test_SOURCES = array_test.cc
random_test_SOURCES = random_test.cc
unicode_test_SOURCES = unicode_test.cc
