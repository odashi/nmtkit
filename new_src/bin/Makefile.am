AM_CXXFLAGS = \
  -I$(top_builddir)/new_src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS) \
  $(CUDA_CPPFLAGS) \
  -I$(top_builddir)/submodules/mteval/src/include \
  -I$(top_builddir)/submodules/spdlog/include

LDADD = \
  -lstdc++ \
  $(top_builddir)/new_src/lib/libnmtkit.la \
  $(BOOST_LDFLAGS) \
  $(BOOST_FILESYSTEM_LIB) \
  $(BOOST_PROGRAM_OPTIONS_LIB) \
  $(BOOST_SERIALIZATION_LIB) \
  $(BOOST_SYSTEM_LIB) \
  $(DYNET_LDFLAGS) \
  $(CUDA_LDFLAGS) \
  $(top_builddir)/submodules/mteval/src/lib/libmteval.la

bin_PROGRAMS = #train decode

#train_SOURCES = train.cc
#train_LDADD = $(LDADD)

#decode_SOURCES = decode.cc
#decode_LDADD = $(LDADD)

