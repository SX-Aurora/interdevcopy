/**
 * @file CUDAMemory.cpp
 * @brief Implementation of CUDA GPU memory type
 */
#include "CUDAMemoryRegion.hpp"
#include "HostMemoryRegion.hpp"
#include "CopyChannel.hpp"

#include "CudaWrapper.hpp"
#include <string>

namespace interdevcopy {
/**
 * @brief Constructor of CUDAMemoryRegion
 * @param p_ pointer to GPU memory
 * @param size size of GPU memory region
 * @param option for future use
 */
CUDAMemoryRegion::CUDAMemoryRegion(void *p_, size_t size,
    __attribute__((unused)) void *option):
  DeviceMemoryRegion(size), devptr(p_) {
    // test if devptr  points GPU memory
    int attr_memory_type;
    CUresult rv = cuda::wrap::cuPointerGetAttribute(&attr_memory_type,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE, cuda::to_deviceptr(p_));
    if (rv != CUDA_SUCCESS) {
      std::string msg("Cannot get pointer attribute ");
      auto rv_str = std::to_string(rv);
      throw std::invalid_argument(msg + rv_str);
    }
    if (attr_memory_type != CU_MEMORYTYPE_DEVICE) {
      throw std::invalid_argument("Not GPU memory");
    }
    // OK. The pointer points GPU memory.
}

namespace between_host_gpu {
/**
 * @brief a function to copy from host to GPU memory
 * @param dst destination GPU memory region
 * @param src source host memory region
 * @param dstoff offset of destination from the start of destination region
 * @param srcoff offset of source from the start of source region
 * @param size the size of area to transfer
 * @param option for future use
 *
 * This function wraps CUDA runtime API function cudaMemcpy()
 * for copy channel to transfer from host to GPU memory.
 */
ssize_t copyFromHostToCUDAMemory(CUDAMemoryRegion *dst, HostMemoryRegion *src,
    size_t dstoff, size_t srcoff, size_t size,
    __attribute__((unused)) void *option) {
  auto rv = cuda::wrap::cudaMemcpy(dst->getPtr(dstoff), src->getPtr(srcoff),
      size, cudaMemcpyHostToDevice);
  INTERDEVCOPY_ASSERT(cudaSuccess == rv);
  return size;
}

/**
 * @brief a function to copy from GPU to host memory
 * @param dst destination host memory region
 * @param src source GPU memory region
 * @param dstoff offset of destination from the start of destination region
 * @param srcoff offset of source from the start of source region
 * @param size the size of area to transfer
 * @param option for future use
 *
 * This function wraps CUDA runtime API function cudaMemcpy()
 * for copy channel to transfer from GPU to host memory.
 */
ssize_t copyFromCUDAMemoryToHost(HostMemoryRegion *dst, CUDAMemoryRegion *src,
    size_t dstoff, size_t srcoff, size_t size,
    __attribute__((unused)) void *option) {
  auto rv = cuda::wrap::cudaMemcpy(dst->getPtr(dstoff), src->getPtr(srcoff),
      size, cudaMemcpyDeviceToHost);
  INTERDEVCOPY_ASSERT(cudaSuccess == rv);
  return size;
}

/**
 * @brief  Copy function factory for transfer from host to GPU memory
 *
 * CopyHostToCUDAMemory defines a copy channel from host to GPU memory.
 */
struct CopyHostToCUDAMemory {
  using srctype = HostMemoryRegion;
  using desttype = CUDAMemoryRegion;
  CopyFuncType operator()(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
      __attribute__((unused)) void *option) {
    return wrapCopyFuncWithDownCastAndBind(&copyFromHostToCUDAMemory,
        dst, src);
  }
};

/**
 * @brief Copy function factory for transfer from GPU to host memory
 *
 * CopyCUDAMemoryToHost defines a copy channel from GPU to host memory.
 */
struct CopyCUDAMemoryToHost {
  using srctype = CUDAMemoryRegion;
  using desttype = HostMemoryRegion;
  CopyFuncType operator()(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
      __attribute__((unused)) void *option) {
    return wrapCopyFuncWithDownCastAndBind(&copyFromCUDAMemoryToHost,
        dst, src);
  }
};
} // namespace interdevcopy::between_host_gpu
} // namespace interdevcopy
REGISTER_DEVICE_MEMORY_REGION_TYPE_IF(interdevcopy::CUDAMemoryRegion,
    interdevcopy::cuda::init_wrapper());

REGISTER_COPY_HANDLER_IF(
    interdevcopy::between_host_gpu::CopyHostToCUDAMemory,
    interdevcopy::cuda::init_wrapper());
REGISTER_COPY_HANDLER_IF(
    interdevcopy::between_host_gpu::CopyCUDAMemoryToHost,
    interdevcopy::cuda::init_wrapper());
