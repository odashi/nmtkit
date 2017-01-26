#ifndef NMTKIT_EXCEPTION_H_
#define NMTKIT_EXCEPTION_H_

#include <stdexcept>
#include <sstream>

#define NMTKIT_FATAL(msg) { \
  std::ostringstream oss; \
  oss << "FATAL: " << __FILE__ << ": " << __LINE__ << ": " << (msg); \
  throw std::runtime_error(oss.str()); \
}

#define NMTKIT_CHECK_MSG(cond, msg) { if (!(cond)) { NMTKIT_FATAL(msg); } }

#define NMTKIT_CHECK(cond) NMTKIT_CHECK_MSG((cond), #cond)

#endif  // NMTKIT_EXCEPTION_H_
