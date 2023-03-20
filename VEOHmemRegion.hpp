/**
 * @file VEOHmemRegion.hpp
 * @brief VE memory (VEO HMEM) type
 */
#ifndef INTERDEVCOPY_VEO_HMEM_REGION_HPP_INCLUDE_
#define INTERDEVCOPY_VEO_HMEM_REGION_HPP_INCLUDE_

#include "DeviceMemoryRegion.hpp"
#include <ve_offload.h>

namespace interdevcopy {
/**
 * @brief VE device memory region
 *
 * Device memory region in VE device memory specified by VEO HMEM address
 */
class VEOHmemRegion: public DeviceMemoryRegion {
  void *hmem;
  veo_proc_handle *proc;
public:
  static constexpr MemoryType memory_type = INTERDEVCOPY_MEMORY_VE_HMEM;
  VEOHmemRegion(void *, size_t, void *);
  veo_proc_handle *getProc() noexcept { return this->proc; }
  void *getHmem(size_t offset = 0) const noexcept {
    return util::addVoidP(this->hmem, offset);
  }
};
} // namespace interdevcopy
#endif
