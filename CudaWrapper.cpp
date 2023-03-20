#include "CudaWrapper.hpp"
#include "Common.hpp"
#include <dlfcn.h>
#include <string>

namespace interdevcopy {
namespace cuda {

std::string _makeErrorMessage(CUresult result, const std::string &msg) {
  return msg + std::string(": ") + std::to_string(result);
}

UnhandleableCUDAError::UnhandleableCUDAError(CUresult result,
    const std::string &msg): runtime_error(_makeErrorMessage(result, msg)) {
}

namespace wrap {
void *libcuda_handle;// a handle of libcuda.so.1
void *libcudart_handle;// a handle of libcudart.so.11

// declarations from <cuda.h>
decltype(::cuPointerGetAttribute) *ptr_cuPointerGetAttribute;
decltype(::cuPointerSetAttribute) *ptr_cuPointerSetAttribute;
decltype(::cudaMemcpy) *ptr_cudaMemcpy;

// wrapper: call a real function in libcuda or libcudart via pointer.

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
    CUdeviceptr ptr)
{
  return ptr_cuPointerGetAttribute(data, attribute, ptr);
}

CUresult cuPointerSetAttribute(const void *value,
    CUpointer_attribute attribute, CUdeviceptr ptr)
{
  return ptr_cuPointerSetAttribute(value, attribute, ptr);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t size,
    enum cudaMemcpyKind kind)
{
  return ptr_cudaMemcpy(dst, src, size, kind);
}

} // namespace interdevcopy::cuda::wrap

#define SET_WRAPPER_(name_, hdl_) do { \
  wrap::ptr_##name_ = reinterpret_cast<decltype(::name_) *>(dlsym(\
        wrap::hdl_, #name_)); \
  if (wrap::ptr_##name_ == nullptr) goto failure; \
} while (0)

#define SET_WRAPPER_D(name_) SET_WRAPPER_(name_, libcuda_handle)
#define SET_WRAPPER_RT(name_) SET_WRAPPER_(name_, libcudart_handle)

/**
 * Initialize CUDA API wrapper
 */
bool init_wrapper() {
  using wrap::libcuda_handle;
  using wrap::libcudart_handle;
  if (libcuda_handle != nullptr && libcudart_handle != nullptr)
    return true;

  libcuda_handle = dlopen("libcuda.so.1", RTLD_LAZY);
  if (libcuda_handle == nullptr) {
    return false;
  }
  // TODO: make CUDA version configurable
  libcudart_handle = dlopen("libcudart.so.11.0", RTLD_LAZY);
  if (libcudart_handle == nullptr) {
    goto failure_libcuda;
  }

  // set wrapper pointers
  SET_WRAPPER_D(cuPointerGetAttribute);
  SET_WRAPPER_D(cuPointerSetAttribute);
  SET_WRAPPER_RT(cudaMemcpy);
  return true;
failure:
  dlclose(libcudart_handle);
failure_libcuda:
  dlclose(libcuda_handle);
  return false;
}

} // namespace interdevcopy::cuda
} // namespace interdevcopy
