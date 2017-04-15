#ifndef NMTKIT_EXCEPTION_H_
#define NMTKIT_EXCEPTION_H_

#include <stdexcept>
#include <sstream>

#define NMTKIT_FATAL(message) { \
  std::ostringstream oss; \
  oss << "FATAL: " << __FILE__ << ": " << __LINE__ << ": " << message; \
  throw std::runtime_error(oss.str()); \
}

#define NMTKIT_CHECK(expr, message) if (!(expr)) { NMTKIT_FATAL(message); }

#define NMTKIT_CHECK_EQ(a, b, message) NMTKIT_CHECK((a) == (b), message)
#define NMTKIT_CHECK_NE(a, b, message) NMTKIT_CHECK((a) != (b), message)

#endif  // NMTKIT_EXCEPTION_H_
