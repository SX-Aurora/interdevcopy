/**
 * @file Assert.cpp
 * @brief Implementation of a function for assertion
 */
#include <cstdio>
#include <cstdlib>

namespace interdevcopy {
namespace util {
/**
 * @brief a helper function for assertion macros
 * @param expr an expression to be tested and fail
 * @param file a file name; __FILE__ is passed.
 * @param line a line number; __LINE__ is passed.
 *
 * Output a message and terminate the process without returning.
 */
[[noreturn]] void _assertion_failure(const char *expr, const char *file,
    int line) {
  fprintf(stderr, "Assertion failure at %s:%d: %s\n", file, line, expr);
  abort();
}

/**
 * @brief a helper function to print an unexpected case due to a bug
 * @param msg a message to output
 * @param file a file name; __FILE__ is passed.
 * @param line a line number; __LINE__ is passed.
 */
[[noreturn]] void _bug(const char *msg, const char *file, int line) {
  fprintf(stderr, "BUG at %s:%d: %s\n", file, line, msg);
  abort();
}
} // namespace interdevcopy::util
} // namespace interdevcopy
