/**
 * @file VEOWrapper.hpp
 * @brief VEO API wrapper
 */
#ifndef INTERDEVCOPY_VEO_WRAPPER_HPP_INCLUDE_
#define INTERDEVCOPY_VEO_WRAPPER_HPP_INCLUDE_

#include <ve_offload.h>
#include <veo_dev_mem.h>
#include <veo_vedma.h>
#include <stdexcept>
#include <string>

namespace interdevcopy {
namespace veo {
bool init_wrapper();

class UnhandleableVEOError: public std::runtime_error {
public:
  UnhandleableVEOError(int, const std::string &);
};

// wrapper functions
namespace wrap {
// ve_offload.h
veo_proc_handle *veo_get_proc_handle_from_hmem(const void *);
// veo_hmem.h
int veo_is_ve_addr(const void *);
// veo_dev_mem.h
int64_t veo_attach_dev_mem(veo_proc_handle *, int);
int veo_detach_dev_mem(veo_proc_handle *, const uint64_t);
int veo_register_gpu_mem(veo_proc_handle *, uint64_t, uint64_t, uint64_t *);
int veo_unregister_gpu_mem(veo_proc_handle *, int);
uint64_t veo_register_mem_to_dmaatb_unalign(void *, size_t);
int veo_unregister_mem_from_dmaatb_unalign(struct veo_proc_handle *, uint64_t);
// veo_vedma.h
int veo_dma_post(struct veo_proc_handle *, uint64_t, uint64_t, int,
    ve_dma_handle_t *);
int veo_dma_poll(struct veo_proc_handle *, ve_dma_handle_t *);
int veo_hmemcpy(void *, const void *, size_t);

} // namespace interdevcopy::veo::wrap
} // namespace interdevcopy::veo
} // namespace interdevcopy
#endif
