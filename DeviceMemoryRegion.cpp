/**
 * @file DeviceMemoryRegion.cpp
 * @brief Implementation of device memory region (base class)
 */

#include "interdevcopy.h"
#include "Common.hpp"
#include "DeviceMemoryRegion.hpp"
#include "Trace.h"

#include <string>
#include <memory>
// std::make_unique() is supported since gcc 4.9.
#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC__MINOR__ < 9)
#include <memory>
namespace std {
template <class T, class... Args> unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
} // namespace std
#endif

namespace {
using interdevcopy::MemoryRegionCtorType;
MemoryRegionCtorType creation_func_table[INTERDEVCOPY_MEMORY_TYPE_MAX];
} // unnamed

namespace interdevcopy {

/**
 * Dummy destructor
 */
DeviceMemoryRegion::~DeviceMemoryRegion(){}

/**
 * @brief register a device memory region type
 * @param t device memory type specified by interdevcopy_memory_type
 * @param c a function to create a device memory region using new operator.
 */
void _registerMemoryRegionConstructor_(MemoryType t, MemoryRegionCtorType c) {
  INTERDEVCOPY_ASSERT(t < INTERDEVCOPY_MEMORY_TYPE_MAX);
  // It is expected that entry has not been set here.
  INTERDEVCOPY_ASSERT(creation_func_table[t] == MemoryRegionCtorType());
  creation_func_table[t] = c;
  INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_DEVICEMEMORY_REGISTER_TYPE(t));
}

/**
 * @brief implementation of interdevcopy_create_memory_region()
 * @param ptr a pointer to device memory
 * @param size the size of device memory region
 * @param type the type of device memory
 * @param option for future use
 *
 * This function implements an interdevcopy API function
 * interdevcopy_create_memory_region().
 * The function finds a creation function (constructor) of device memory
 * region for specified type and create using the creation function.
 */
interdevcopy_memory_region *createMemoryRegion_(void *ptr, size_t size,
    MemoryType type, void *option) {
  if (ptr == nullptr) {
    std::string msg("invalid memory pointer ");
    throw std::invalid_argument(msg);
  }
  if (type >= INTERDEVCOPY_MEMORY_TYPE_MAX) {
    std::string msg("memory type out of range ");
    auto memory_type_str = std::to_string(type);
    throw std::invalid_argument(msg + memory_type_str);
  }
  auto ctor = creation_func_table[type];
  if (ctor == nullptr) {
    std::string msg("invalid memory type ");
    auto memory_type_str = std::to_string(type);
    throw std::invalid_argument(msg + memory_type_str);
  }
  // unique_ptr is used to avoid memory leak on an exception in ctor.
  auto dm = std::make_unique<interdevcopy_memory_region>();
  auto implp = (*ctor)(ptr, size, option);
  dm->implp = implp;
  if (INTERDEVCOPY_TRACE_ENABLED(LIBINTERDEVCOPY_DEVICEMEMORY_CREATE)) {
    auto name = util::demangle(typeid(*implp).name());
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_DEVICEMEMORY_CREATE(ptr, size,
          name.c_str(), implp));
  }
  auto rv = dm.release();
  return rv;

}

/**
 * @brief implementation of interdevcopy_destroy_memory_region()
 * @param mem device memory region
 *
 * This function implements an interdevcopy API function
 * interdevcopy_destroy_memory_region().
 */
int destroyMemoryRegion_(interdevcopy_memory_region *mem) {
  if (mem == nullptr) {
    std::string msg("invalid memory pointer ");
    throw std::invalid_argument(msg);
  }
  auto devmem = mem->implp;
  if (devmem->isUsed())
    throw std::system_error(std::make_error_code(
          std::errc::device_or_resource_busy));
  delete devmem;
  delete mem;
  return 0;
}
} // namespace interdevcopy

/* API functions */
/**
 * @ingroup API
 * @brief API function to create a device memory region
 * @param ptr a pointer to device memory
 * @param size the size of device memory region
 * @param type the type of device memory
 * @param option for future use
 * @return pointer to memory region upon success; negative upon failure.
 */
interdevcopy_memory_region *interdevcopy_create_memory_region(void *ptr,
    size_t size, enum interdevcopy_memory_type type, void *option) {
  INTERDEVCOPY_API_WRAPPER(interdevcopy::createMemoryRegion_(
        ptr, size, type, option));
}

/**
 * @ingroup API
 * @brief API function to destroy a device memory region
 * @param mem device memory region
 * @return zero upon success; negative upon failure.
 */
int interdevcopy_destroy_memory_region(interdevcopy_memory_region *mem) {
  INTERDEVCOPY_API_WRAPPER(interdevcopy::destroyMemoryRegion_(mem));
}
