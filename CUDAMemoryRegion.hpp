/**
 * @file CUDAMemoryRegion.hpp
 * @brief CUDA GPU memory type
 */
#ifndef INTERDEVCOPY_CUDA_MEMORY_REGION_HPP_INCLUDE_
#define INTERDEVCOPY_CUDA_MEMORY_REGION_HPP_INCLUDE_

#include "DeviceMemoryRegion.hpp"

namespace interdevcopy {
/**
 * @brief CUDA GPU device memory region
 *
 * Device memory region in CUDA GPU device memory
 */
class CUDAMemoryRegion: public DeviceMemoryRegion {
  void *devptr;
public:
  static constexpr MemoryType memory_type = INTERDEVCOPY_MEMORY_CUDA_MEM;
  CUDAMemoryRegion(void *, size_t, void *);
  /**
   * @brief GPU memory pointer
   * @param offset offset from the start of the memory region.
   * @return pointer to CUDA memory
   */
  void *getPtr(size_t offset = 0) const noexcept {
    return util::addVoidP(this->devptr, offset);
  }
};
}
#endif
