/**
 * @file HostMemory.cpp
 * @brief Implementation of host memory type
 */
#include "HostMemoryRegion.hpp"
#include "CopyChannel.hpp"
#include <cstring>

namespace interdevcopy {
/**
 * @brief a function to copy between host memory
 * @param dst destination host memory region
 * @param src source host memory region
 * @param dstoff offset of destination from the start of destination region
 * @param srcoff offset of source from the start of source region
 * @param size the size of area to transfer
 * @param option for future use
 */
ssize_t copyFromHostToHost(HostMemoryRegion *dst, HostMemoryRegion *src,
    size_t dstoff, size_t srcoff, size_t size,
    __attribute__((unused)) void *option) {
  memmove(dst->getPtr(dstoff), src->getPtr(srcoff), size);
  return size;
}
/**
 * @brief Copy function factory for transfer between host memory
 *
 * CopyHostToThst defines a copy channel between host memory.
 */
struct CopyHostToHost {
  using srctype = HostMemoryRegion;
  using desttype = HostMemoryRegion;
  CopyFuncType operator()(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
      __attribute__((unused)) void *option) {
    return wrapCopyFuncWithDownCastAndBind(&copyFromHostToHost, dst, src);
  }
};
}
REGISTER_DEVICE_MEMORY_REGION_TYPE(interdevcopy::HostMemoryRegion);
REGISTER_COPY_HANDLER(interdevcopy::CopyHostToHost);
