/**
 * @file Demangle.cpp
 * @brief Implementation of demangle function
 */

#include "config.h"
#include "Common.hpp"
#ifdef HAVE_CXXABI_H
#include <cxxabi.h>
#endif

namespace interdevcopy {
namespace util {

/**
 * @brief demangle name
 * @param name C++ mangled symbol name
 * @return demangled name
 *
 * This function demangles a symbol name using libstdc++ runtime library.
 * Probes use demangle() to output object types.
 */
std::string demangle(const char *name) {
#ifdef HAVE_CXXABI_H
  int status;
  auto demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  std::string rv(demangled);
  INTERDEVCOPY_ASSERT(demangled != nullptr);
  INTERDEVCOPY_ASSERT(status == 0);
  free(demangled);
#else
  std::string rv(name);
#endif
  return rv;
}
} // namespace interdevcopy::util
} // namespace interdevcopy

