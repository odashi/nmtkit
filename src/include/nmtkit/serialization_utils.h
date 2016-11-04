#ifndef NMTKIT_SERIALIZATION_UTILS_H_
#define NMTKIT_SERIALIZATION_UTILS_H_

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>

#define NMTKIT_SERIALIZATION_DECL(cls) BOOST_CLASS_EXPORT_KEY(cls)

#define NMTKIT_SERIALIZATION_IMPL(cls) \
  BOOST_CLASS_EXPORT_IMPLEMENT(cls) \
  template void cls::serialize( \
      boost::archive::text_iarchive &, const unsigned); \
  template void cls::serialize( \
      boost::archive::text_oarchive &, const unsigned);

#endif  // NMTKIT_SERIALIZATION_UTILS_H_
