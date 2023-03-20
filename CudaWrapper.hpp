/**
 * @file CudaWrapper.hpp
 * @brief CUDA API wrapper
 */
#ifndef INTERDEVCOPY_CUDA_WRAPPER_HPP_INCLUDE_
#define INTERDEVCOPY_CUDA_WRAPPER_HPP_INCLUDE_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace interdevcopy {
namespace cuda {
bool init_wrapper();

class UnhandleableCUDAError: public std::runtime_error {
public:
  UnhandleableCUDAError(CUresult, const std::string &);
};

// wrapper funcitons
namespace wrap {
CUresult cuPointerGetAttribute(void *, CUpointer_attribute, CUdeviceptr);
CUresult cuPointerSetAttribute(const void *, CUpointer_attribute, CUdeviceptr);

cudaError_t cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind);

} // namespace interdevcopy::cuda::wrap

template <typename T> CUdeviceptr to_deviceptr(T *p)
{
  return reinterpret_cast<CUdeviceptr>(p);
}
} // namespace interdevcopy::cuda
} // namespace interdevcopy
#endif
