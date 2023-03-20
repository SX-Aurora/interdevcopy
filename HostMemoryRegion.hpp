/**
 * @file HostMemoryRegion.hpp
 * @brief Host memory type
 */
#ifndef INTERDEVCOPY_HOST_MEMORY_REGION_HPP_INCLUDE_
#define INTERDEVCOPY_HOST_MEMORY_REGION_HPP_INCLUDE_

#include "DeviceMemoryRegion.hpp"

namespace interdevcopy {
/**
 * @brief Host memory region
 */
class HostMemoryRegion: public DeviceMemoryRegion {
  void *ptr;
public:
  static constexpr MemoryType memory_type = INTERDEVCOPY_MEMORY_HOST_MEM;
  HostMemoryRegion(void *p_, size_t size,
      __attribute__((unused)) void *option):
    DeviceMemoryRegion(size),
    ptr(p_) {}
  /**
   * @brief Host memory pointer
   * @param offset offset from the start of the memory region
   * @return pointer to host memory
   */
  void *getPtr(size_t offset) const noexcept {
    // without boundary check; call after confirming offset < size.
    return util::addVoidP(this->ptr, offset);
  }
};
}
#endif
