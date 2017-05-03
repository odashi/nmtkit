#ifndef NMTKIT_LOGGER_H_
#define NMTKIT_LOGGER_H_

#include <fstream>
#include <string>
#include <boost/format.hpp>
#include <nmtkit/exception.h>

namespace nmtkit {

// Simple logging interface for single value.
class Logger {
  Logger() = delete;
  Logger(const Logger &) = delete;
  Logger(Logger &&) = delete;
  Logger & operator=(const Logger &) = delete;
  Logger & operator=(Logger &&) = delete;

public:
  // Create a logger object.
  //
  // Arguments:
  //   filepath: Location of the log file.
  //   format: Formatting string. See boost::format documentation.
  Logger(const std::string & filepath, const std::string & format)
  : ofs_(filepath), fmt_(format) {
    NMTKIT_CHECK(ofs_.is_open(), "Unable to open file to write: " + filepath);
  }

  ~Logger() {}

  // Log a value.
  //
  // Arguments:
  //   value: Value to log.
  template <typename T>
  void log(const T & value) {
    ofs_ << (boost::format(fmt_) % value) << std::endl;
  }

private:
  std::ofstream ofs_;
  std::string fmt_;
};

}  // namespace nmtkit

#endif // NMTKIT_LOGGER_H_
