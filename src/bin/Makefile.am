AM_CXXFLAGS = -I$(srcdir)/../include $(BOOST_CPPFLAGS)
LDADD = -lstdc++ ../lib/libnmtkit.la $(BOOST_LDFLAGS) $(BOOST_FILESYSTEM_LIB) $(BOOST_PROGRAM_OPTIONS_LIB) $(BOOST_SYSTEM_LIB)

bin_PROGRAMS = train

train_SOURCES = train.cc
train_LDADD = $(LDADD)

