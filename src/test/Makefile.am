AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS)
LDADD = -lstdc++ ../lib/libnmtkit.la $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

check_PROGRAMS = \
  monotone_sampler_test \
  random_test \
  random_sampler_test \
	vocabulary_test

TESTS = \
  monotone_sampler_test \
  random_test \
  random_sampler_test \
	vocabulary_test

monotone_sampler_test_SOURCES = monotone_sampler_test.cc
random_test_SOURCES = random_test.cc
random_sampler_test_SOURCES = random_sampler_test.cc
vocabulary_test_SOURCES = vocabulary_test.cc
