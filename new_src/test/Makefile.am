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
  character_vocabulary_test \
  random_test \
  simple_text_reader_test \
  unicode_test \
  word_vocabulary_test

TESTS = \
  array_test \
  character_vocabulary_test \
  random_test \
  simple_text_reader_test \
  unicode_test \
  word_vocabulary_test

array_test_SOURCES = array_test.cc
character_vocabulary_test_SOURCES = character_vocabulary_test.cc
random_test_SOURCES = random_test.cc
simple_text_reader_test_SOURCES = simple_text_reader_test.cc
unicode_test_SOURCES = unicode_test.cc
word_vocabulary_test_SOURCES = word_vocabulary_test.cc
