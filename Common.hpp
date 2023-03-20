/**
 * @file Common.hpp
 * @brief common definitions for internal use
 */
#ifndef INTERDEVCOPY_COMMON_HPP_INCLUDE_
#define INTERDEVCOPY_COMMON_HPP_INCLUDE_

#include "interdevcopy.h"
#include "Assert.hpp"

#include <utility>
#include <exception>
#include <stdexcept>
#include <system_error>
#include <functional>
#include <type_traits>
#include <string>
#include <cstddef>

namespace interdevcopy {
using MemoryType = ::interdevcopy_memory_type;

namespace util {

// Cast()
/**
 * @brief Safe cast between integer and pointer types
 * @tparam ToT type casted to
 * @tparam FromT type casted from
 * @param from a value to cast
 * @return the value casted from FromT to ToT.
 */
template <typename ToT, typename FromT>
typename std::enable_if<std::is_same<ToT, FromT>::value, ToT>::type
Cast(FromT from) {
  // when FromT and ToT are same, no cast is necessary.
    return from;
}

template <typename ToT, typename FromT>
typename std::enable_if<
  std::is_integral<ToT>::value && std::is_integral<FromT>::value &&
  !std::is_same<ToT, FromT>::value, ToT
>::type Cast(FromT from) {
  // Casting from integer to integer, static cast works
  return static_cast<ToT>(from);
}

template <typename ToT, typename FromT>
typename std::enable_if<
  std::is_pointer<ToT>::value && !std::is_pointer<FromT>::value &&
  sizeof(ToT) >= sizeof(FromT), ToT
>::type Cast(FromT from) {
  // Casting from integer to pointer,
  // extend the value to intptr_t and reinterpret as pointer.
  return reinterpret_cast<ToT>(static_cast<intptr_t>(from));
}

template<typename ToT, typename FromT>
typename std::enable_if<
  std::is_integral<ToT>::value && std::is_pointer<FromT>::value &&
  sizeof(ToT) == sizeof(FromT), ToT
>::type Cast(FromT from) {
  // Casting from pointer to integer,
  // ToT is required to have the same width as pointer.
  return reinterpret_cast<ToT>(from);
}


/**
 * @brief a wrapper for C API to avoid exceptions
 * @param expr_ an expression which can throw exceptions.
 *
 * Translate an exception into the corresponding error number.
 * This macro evaluates an expression and execute return statement
 * if no exceptions are caught; if an exception is thrown,
 * execute a return statement with the negative of error code
 * corresponding to the exception caught.
 */
#define INTERDEVCOPY_API_WRAPPER(expr_) do { \
  using RetType_ = decltype(expr_);\
  try { \
    return (expr_); \
  } catch (std::system_error &e) { \
    return interdevcopy::util::Cast<RetType_>(-e.code().value()); \
  } catch (std::invalid_argument &e) { \
    return interdevcopy::util::Cast<RetType_>(-EINVAL); \
  } catch (std::bad_alloc &e) { \
    return interdevcopy::util::Cast<RetType_>(-ENOMEM); \
  } catch (...) { \
    interdevcopy::util::_bug("unexpected exception was thrown.", \
        __FILE__, __LINE__); \
  } \
} while (0)

/**
 * @brief An internal function for errnoToException().
 * @param e standard error number
 * @param what explanatory information
 *
 * Throw an exception corresponding to a standard error code specified
 * by the argument e.
 */
[[noreturn]] inline void _errnoToException(int e, const char *what) {
  throw std::system_error(e, std::generic_category(), what);
}

/**
 * @brief Translate an error code into an exception
 * @param rv a value
 * @param what explanatory information
 * @return rv, or the passed value, unless the value is negative.
 *
 * If a non-negative value is passed, return the value.
 * if the value is negative, throw an exception as the value is
 * a negative of standard error code.
 */
template <typename T> typename std::enable_if<
  std::is_integral<T>::value && std::is_signed<T>::value, T
>::type errnoToException(T rv, const char *what) {
  if (rv >= 0) return rv;
  _errnoToException(-rv, what);
}

/**
 * @brief add operation to void pointer
 * @param p a void pointer
 * @param off offset of data from p
 * @return a pointer of the position with the offset off from p
 *
 * The function wraps an add operation to a void pointer with a messy cast.
 */
constexpr void *addVoidP(void *p, ptrdiff_t off) noexcept {
  return static_cast<char *>(p) + off;
}

/**
 * @brief a base class of libinterdevcopy internal objects
 *
 * A base class for non-copyable objects with usage counter.
 */
class Object {
  unsigned int refcount;
public:
  // non-copyable
  Object(const Object &) = delete;
  Object &operator=(const Object &) = delete;
  Object(): refcount(0) {}
  /**
   * @brief increment the usage count.
   */
  void get() noexcept {
    ++this->refcount;
    INTERDEVCOPY_ASSERT(this->refcount > 0);
  }
  /**
   * @brief decrement the usage count.
   */
  void put() noexcept {
    INTERDEVCOPY_ASSERT(this->refcount > 0);
    --this->refcount;
  }
  /**
   * @brief test the object is used
   * @retval true the object is used.
   * @retval false the object is not used.
   *
   * The method returns true if the usage count of object is non-zero;
   * returns false if the usage count is positive.
   */
  bool isUsed() const noexcept { return this->refcount > 0; }
};

/**
 * @brief a utility class to execute a procedure on exiting a scope
 *
 * Set a procedure which is required to run finally, and it is executed
 * at the exit of scope by RAII.
 */
class Finally {
  std::function<void()> procedure;

public:
  Finally(const Finally &) = delete;
  Finally &operator=(const Finally &) = delete;
  explicit Finally(std::function<void()> f): procedure(f) {}
  ~Finally() { this->procedure(); }
};

std::string demangle(const char *name);
} // namespace interdevcopy::util
} // namespace interdevcopy

#define _INTERDEVCOPY_CONCAT_IMPL__(a_, b_) a_ ## b_
#define _INTERDEVCOPY_CONCAT__(a_, b_) _INTERDEVCOPY_CONCAT_IMPL__(a_, b_)
#define _INTERDEVCOPY_UNIQUE_SYMBOL__(name_) \
  _INTERDEVCOPY_CONCAT__(name_, __LINE__)
#endif
