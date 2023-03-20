/**
 * @file Device MemoryRegion.hpp
 * @brief Device memory region header
 */
#ifndef INTERDEVCOPY_DEVICE_MEMORY_REGION_HPP_INCLUDE_
#define INTERDEVCOPY_DEVICE_MEMORY_REGION_HPP_INCLUDE_

#include "Common.hpp"

namespace interdevcopy {

/**
 * @brief device memory region
 *
 * The base claass of device memory regions with the size of region.
 * A derived class for a specific device memory type is supposed to
 * implement a method to get a pointer to memory area.
 */
class DeviceMemoryRegion: public util::Object {
  size_t size;
public:
  DeviceMemoryRegion(size_t sz): size(sz) {}
  virtual ~DeviceMemoryRegion() = 0;
  size_t getSize() const noexcept { return this->size; }
};

using MemoryRegionCtorType = DeviceMemoryRegion *(*)(void *, size_t, void *);
void _registerMemoryRegionConstructor_(MemoryType, MemoryRegionCtorType);

/**
 * @brief template class for registering device meory region type
 * @tparam DevMemType class of device memory region
 *
 * The template class _DeviceMemoryRegionTypeInitializer registers a
 * the type of device memory region on initialization.
 */
template <typename DevMemType>
struct _DeviceMemoryRegionTypeInitializer {
  /**
   * @brief constructor to run a routine on initialization
   * @param cond condition expression
   *
   * Register a device memory region type if the condition is true.
   */
  _DeviceMemoryRegionTypeInitializer(std::function<bool()> &&cond) {
    if (cond()) {
      _registerMemoryRegionConstructor_(DevMemType::memory_type,
          [](void *p_, size_t sz_, void *opt_)->DeviceMemoryRegion * {
        return new DevMemType(p_, sz_, opt_);
      });
    }
  }
};
} // namespace interdevcopy
/**
 * @ingroup API
 * @brief device memory region(exported data type as part of API)
 * @see interdevcopy_create_memory_region
 */
struct interdevcopy_memory_region {
  interdevcopy::DeviceMemoryRegion *implp;
};
/**
 * @var interdevcopy_memory_region::implp
 * @brief use a pointer to save derived type information
 */

/**
 * @brief register a device memory region type
 * @param T_ type name of device memory region class.
 *        The class is supposed to be derived from DeviceMemoryRegion.
 * @param condexpr_ a condition expression
 */
#define REGISTER_DEVICE_MEMORY_REGION_TYPE_IF(T_, condexpr_) \
namespace { \
  interdevcopy::_DeviceMemoryRegionTypeInitializer<T_> \
    _INTERDEVCOPY_UNIQUE_SYMBOL__(_dev_mem_type_init_)(\
        [](){ return (condexpr_); }); \
}
#define REGISTER_DEVICE_MEMORY_REGION_TYPE(T_) \
  REGISTER_DEVICE_MEMORY_REGION_TYPE_IF(T_, true)
#endif
