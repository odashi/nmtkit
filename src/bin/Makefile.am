AM_CXXFLAGS = \
  -I$(top_builddir)/src/include \
  $(BOOST_CPPFLAGS) \
  $(EIGEN_CPPFLAGS) \
  $(DYNET_CPPFLAGS) \
  -I$(top_builddir)/submodules/mteval/src/include

LDADD = \
  -lstdc++ \
  $(top_builddir)/src/lib/libnmtkit.la \
  $(BOOST_LDFLAGS) \
  $(BOOST_FILESYSTEM_LIB) \
  $(BOOST_PROGRAM_OPTIONS_LIB) \
  $(BOOST_SERIALIZATION_LIB) \
  $(BOOST_SYSTEM_LIB) \
  $(DYNET_LDFLAGS) \
  $(CUDA_LDFLAGS) \
  $(top_builddir)/submodules/mteval/src/lib/libmteval.la

bin_PROGRAMS = train decode

train_SOURCES = train.cc
train_LDADD = $(LDADD)

decode_SOURCES = decode.cc
decode_LDADD = $(LDADD)

