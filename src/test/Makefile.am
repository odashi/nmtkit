AM_CXXFLAGS = \
  -I$(top_builddir)/src/include \
  $(BOOST_CPPFLAGS)

LDADD = \
  -lstdc++ \
  $(top_builddir)/src/lib/libnmtkit.la \
  $(BOOST_UNIT_TEST_FRAMEWORK_LIB) \
  $(DYNET_LDFLAGS)

check_PROGRAMS = \
  array_test \
  batch_converter_test \
  corpus_test \
  monotone_sampler_test \
  random_test \
  random_sampler_test \
  sorted_random_sampler_test \
	vocabulary_test

TESTS = \
  array_test \
  batch_converter_test \
  corpus_test \
  monotone_sampler_test \
  random_test \
  random_sampler_test \
  sorted_random_sampler_test \
	vocabulary_test

array_test_SOURCES = array_test.cc
batch_converter_test_SOURCES = batch_converter_test.cc
corpus_test_SOURCES = corpus_test.cc
monotone_sampler_test_SOURCES = monotone_sampler_test.cc
random_test_SOURCES = random_test.cc
random_sampler_test_SOURCES = random_sampler_test.cc
sorted_random_sampler_test_SOURCES = sorted_random_sampler_test.cc
vocabulary_test_SOURCES = vocabulary_test.cc
