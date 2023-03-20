/**
 * @file Assert.hpp
 * @brief Assertion macros
 */
#ifndef INTERDEVCOPY_ASSERT_HPP_INCLUDE_
#define INTERDEVCOPY_ASSERT_HPP_INCLUDE_

#include <cstdio>
namespace interdevcopy {
namespace util {
[[noreturn]] void _assertion_failure(const char *, const char *, int);
[[noreturn]] void _bug(const char *, const char *, int);
}// namespace interdevcopy::util
}// namespace interdevcopy

/**
 * @def INTERDEVCOPY_ASSERT(expr_)
 * @brief a basic assertion macro
 * @param expr_ an expression to test
 *
 * Prints an error message and terminates the program if expr is false.
 */
#define INTERDEVCOPY_ASSERT(expr_) \
  do { if (!(expr_)) \
        interdevcopy::util::_assertion_failure(#expr_, __FILE__, __LINE__); \
  } while (0)

/**
 * @def INTERDEVCOPY_ASSERT_ZERO(expr_)
 * @brief an assertion macro to test the expression is zero.
 * @param expr_ an expression to test
 *
 * Prints an error message and the (unexpected) value of expression,
 * and terminates the program if expr is not zero.
 */
#define INTERDEVCOPY_ASSERT_ZERO(expr_) \
  do { long val_ = (expr_); \
    if (val_ != 0) { \
      char msgstr_[sizeof(#expr_)+64]; \
      std::snprintf(msgstr_, sizeof(msgstr_), "%s is not 0 (%ld)", \
                    #expr_, val_); \
      interdevcopy::util::_assertion_failure(msgstr_, __FILE__, __LINE__); \
    } \
  } while (0)
#endif
