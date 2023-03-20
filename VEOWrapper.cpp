#include "VEOWrapper.hpp"
#include "Common.hpp"

#include <dlfcn.h>
#include <string>


namespace interdevcopy {
namespace veo {

std::string _makeErrorMessage(int result, const std::string &msg) {
  return msg + std::string(": ") + std::to_string(result);
}
UnhandleableVEOError::UnhandleableVEOError(int result,
    const std::string &msg): runtime_error(_makeErrorMessage(result, msg)) {
}
namespace wrap {
void *libveo_handle;// a handle of libveo.so.1

// declarations of pointers to functions in libveo
#define DECLARE_FUNC_PTR(funcname_)  decltype(:: funcname_) *ptr_ ## funcname_
DECLARE_FUNC_PTR(veo_get_proc_handle_from_hmem);
DECLARE_FUNC_PTR(veo_is_ve_addr);
DECLARE_FUNC_PTR(veo_attach_dev_mem);
DECLARE_FUNC_PTR(veo_detach_dev_mem);
DECLARE_FUNC_PTR(veo_register_gpu_mem);
DECLARE_FUNC_PTR(veo_unregister_gpu_mem);
DECLARE_FUNC_PTR(veo_register_mem_to_dmaatb_unalign);
DECLARE_FUNC_PTR(veo_unregister_mem_from_dmaatb_unalign);
DECLARE_FUNC_PTR(veo_dma_post);
DECLARE_FUNC_PTR(veo_dma_poll);
DECLARE_FUNC_PTR(veo_hmemcpy);

// wrapper functions
veo_proc_handle *veo_get_proc_handle_from_hmem(const void *hmem) {
  return ptr_veo_get_proc_handle_from_hmem(hmem);
}

int veo_is_ve_addr(const void *hmem) {
  return ptr_veo_is_ve_addr(hmem);
}

int64_t veo_attach_dev_mem(veo_proc_handle *h, int devmem_id) {
  return ptr_veo_attach_dev_mem(h, devmem_id);
}

int veo_detach_dev_mem(veo_proc_handle *h, const uint64_t vehva) {
  return ptr_veo_detach_dev_mem(h, vehva);
}

int veo_register_gpu_mem(veo_proc_handle *h, uint64_t vaddr, uint64_t size,
    uint64_t *reg_size) {
  return ptr_veo_register_gpu_mem(h, vaddr, size, reg_size);
}

int veo_unregister_gpu_mem(veo_proc_handle *h, int devmem_id) {
  return ptr_veo_unregister_gpu_mem(h, devmem_id);
}

uint64_t veo_register_mem_to_dmaatb_unalign(void *hmem, uint64_t size) {
  return ptr_veo_register_mem_to_dmaatb_unalign(hmem, size);
}

int veo_unregister_mem_from_dmaatb_unalign(veo_proc_handle *proc,
    uint64_t size) {
  return ptr_veo_unregister_mem_from_dmaatb_unalign(proc, size);
}

int veo_dma_post(veo_proc_handle *proc, uint64_t dst, uint64_t src, int size,
    ve_dma_handle_t *hdlp) {
  return ptr_veo_dma_post(proc, dst, src, size, hdlp);
}

int veo_dma_poll(veo_proc_handle *proc, ve_dma_handle_t *hdlp) {
  return ptr_veo_dma_poll(proc, hdlp);
}

int veo_hmemcpy(void *dst, const void *src, size_t size) {
  return ptr_veo_hmemcpy(dst, src, size);
}

} // namespace interdevcopy::veo::wrap

#define SET_WRAPPER(name_) do { \
  wrap::ptr_##name_ = reinterpret_cast<decltype(::name_) *>(dlsym( \
        wrap::libveo_handle, #name_)); \
  if (wrap::ptr_##name_ == nullptr) goto failure; \
} while (0)

/**
 * Initialize VEO API wrapper
 */
bool init_wrapper() {
  using wrap::libveo_handle;
  if (libveo_handle != nullptr)
    return true;

  libveo_handle = dlopen("libveo.so.1", RTLD_LAZY);
  if (libveo_handle == nullptr) {
    return false;
  }

  // set wrapper pointers
  SET_WRAPPER(veo_get_proc_handle_from_hmem);
  SET_WRAPPER(veo_is_ve_addr);
  SET_WRAPPER(veo_attach_dev_mem);
  SET_WRAPPER(veo_detach_dev_mem);
  SET_WRAPPER(veo_register_gpu_mem);
  SET_WRAPPER(veo_unregister_gpu_mem);
  SET_WRAPPER(veo_register_mem_to_dmaatb_unalign);
  SET_WRAPPER(veo_unregister_mem_from_dmaatb_unalign);
  SET_WRAPPER(veo_dma_post);
  SET_WRAPPER(veo_dma_poll);
  SET_WRAPPER(veo_hmemcpy);

  return true;
failure:
  dlclose(libveo_handle);
  return false;
}

} // namespace interdevcopy::veo
} // namespace interdevcopy
