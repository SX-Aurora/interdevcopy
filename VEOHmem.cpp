/**
 * @file VEOHmem.cpp
 * @brief Implementation of VEO HMEM type
 */
#include "VEOHmemRegion.hpp"
#include "HostMemoryRegion.hpp"
#include "CopyChannel.hpp"

#include "VEOWrapper.hpp"

namespace interdevcopy {
/**
 * @brief Constructor of VEOHmemRegion
 * @param p_ VEO heterogeneous memory, an VE address with process identifier
 * @param size size of VE memory region
 * @param option for future use
 */
VEOHmemRegion::VEOHmemRegion(void *p_, size_t size,
    __attribute__((unused)) void *option):
  DeviceMemoryRegion(size), hmem(p_) {
    // test if hmem points VE memory
    if (!veo::wrap::veo_is_ve_addr(this->hmem)) {
      throw std::invalid_argument("Not VE memory");
    }
    this->proc = veo::wrap::veo_get_proc_handle_from_hmem(this->hmem);
    INTERDEVCOPY_ASSERT(this->proc != nullptr);
    // OK. The pointer points VE memory
}

namespace between_host_ve {
/**
 * @brief a function to copy from host to VE memory
 * @param dst destination VEO hmem region
 * @param src source host memory region
 * @param dstoff offset of destination from the start of destination region
 * @param srcoff offset of source from the start of source region
 * @param size the size of area to transfer
 * @param option for future use
 *
 * This function wraps VEO API function veo_hmemcpy()
 * for copy channel to transfer from host to VE memory.
 */
ssize_t copyFromHostToVEOHmem(VEOHmemRegion *dst, HostMemoryRegion *src,
    size_t dstoff, size_t srcoff, size_t size,
    __attribute__((unused)) void *option) {
  INTERDEVCOPY_ASSERT_ZERO(veo::wrap::veo_hmemcpy(dst->getHmem(dstoff),
        src->getPtr(srcoff), size));
  return size;
}

/**
 * @brief a function to copy from VE to host memory
 * @param dst destination host memory region
 * @param src source VEO hmem region
 * @param dstoff offset of destination from the start of destination region
 * @param srcoff offset of source from the start of source region
 * @param size the size of area to transfer
 * @param option for future use
 *
 * This function wraps VEO API function veo_hmemcpy()
 * for copy channel to transfer from VE to host memory.
 */
ssize_t copyFromVEOHmemToHost(HostMemoryRegion *dst, VEOHmemRegion *src,
    size_t dstoff, size_t srcoff, size_t size,
    __attribute__((unused)) void *option) {
  INTERDEVCOPY_ASSERT_ZERO(veo::wrap::veo_hmemcpy(dst->getPtr(dstoff),
        src->getHmem(srcoff), size));
  return size;
}

/**
 * @brief Copy function actory for transfer from host to VE memory
 *
 * CopyHostToVEOHmem defines a copy channel from host to VE memory.
 */
struct CopyHostToVEOHmem {
  using srctype = HostMemoryRegion;
  using desttype = VEOHmemRegion;
  CopyFuncType operator()(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
      __attribute__((unused)) void *option) {
    return wrapCopyFuncWithDownCastAndBind(&copyFromHostToVEOHmem,
        dst, src);
  }
};

/**
 * @brief Copy function actory for transfer from VE to host memory
 *
 * CopyHostToVEOHmem defines a copy channel from VE to host memory.
 */
struct CopyVEOHmemToHost {
  using srctype = VEOHmemRegion;
  using desttype = HostMemoryRegion;
  CopyFuncType operator()(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
      __attribute__((unused)) void *option) {
    return wrapCopyFuncWithDownCastAndBind(&copyFromVEOHmemToHost,
        dst, src);
  }
};
} // namespace interdevcopy::between_host_ve
} // namespace interdevcopy
REGISTER_DEVICE_MEMORY_REGION_TYPE_IF(interdevcopy::VEOHmemRegion,
    interdevcopy::veo::init_wrapper());

REGISTER_COPY_HANDLER_IF(
    interdevcopy::between_host_ve::CopyHostToVEOHmem,
    interdevcopy::veo::init_wrapper());
REGISTER_COPY_HANDLER_IF(
    interdevcopy::between_host_ve::CopyVEOHmemToHost,
    interdevcopy::veo::init_wrapper());
