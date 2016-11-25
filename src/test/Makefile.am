AM_CXXFLAGS = \
  -I$(top_builddir)/src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS) \
  $(CUDA_CPPFLAGS)

LDADD = \
  -lstdc++ \
  $(top_builddir)/src/lib/libnmtkit.la \
  $(BOOST_UNIT_TEST_FRAMEWORK_LIB) \
  $(BOOST_SERIALIZATION_LIB) \
  $(DYNET_LDFLAGS) \
  $(CUDA_LDFLAGS)

check_PROGRAMS = \
  array_test \
  batch_converter_test \
  character_vocabulary_test \
  corpus_test \
  inference_graph_test \
  monotone_sampler_test \
  random_test \
  sorted_random_sampler_test \
	word_vocabulary_test

TESTS = \
  array_test \
  batch_converter_test \
  character_vocabulary_test \
  corpus_test \
  inference_graph_test \
  monotone_sampler_test \
  random_test \
  sorted_random_sampler_test \
	word_vocabulary_test

array_test_SOURCES = array_test.cc
batch_converter_test_SOURCES = batch_converter_test.cc
character_vocabulary_test_SOURCES = character_vocabulary_test.cc
corpus_test_SOURCES = corpus_test.cc
inference_graph_test_SOURCES = inference_graph_test.cc
monotone_sampler_test_SOURCES = monotone_sampler_test.cc
random_test_SOURCES = random_test.cc
sorted_random_sampler_test_SOURCES = sorted_random_sampler_test.cc
word_vocabulary_test_SOURCES = word_vocabulary_test.cc
