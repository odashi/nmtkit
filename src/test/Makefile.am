AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS)
LDADD = -lstdc++ ../lib/libnmtkit.la $(BOOST_UNIT_TEST_FRAMEWORK_LIB)

check_PROGRAMS = \
	train_test

TESTS = \
	train_test

train_test_SOURCES = train_test.cc
