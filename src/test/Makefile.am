AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS)
LDADD = -lstdc++ ../lib/libnmtkit.la $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

check_PROGRAMS = \
  sampler_test \
	vocabulary_test

TESTS = \
  sampler_test \
	vocabulary_test

sampler_test_SOURCES = sampler_test.cc
vocabulary_test_SOURCES = vocabulary_test.cc
