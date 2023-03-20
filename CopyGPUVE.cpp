/**
 * @file CopyGPUVE.cpp
 * @brief Implementation of transfer betwen CUDA GPU and VE
 */

#include "CopyChannel.hpp"
#include "CUDAMemoryRegion.hpp"
#include "CudaWrapper.hpp"
#include "VEOHmemRegion.hpp"
#include "VEOWrapper.hpp"
#include "Trace.h"

#include <algorithm>
#include <list>
#include <memory>
#include <utility>
#include <sstream>

namespace interdevcopy {
namespace between_gpu_ve {

/// Maximum size of a request VE DMA engine supports.
constexpr size_t MAX_DMA_REQUEST_SIZE = 128 * (1 << 20) - 4;

/**
 * desccriptor of a pair of contiguous area of GPU and VE memory to transfer
 */
struct DMAAddrArea {
  uint64_t vehva_gpumem;/// VEHVA of GPU memory
  uint64_t vehva_vemem;/// VEHVA of VE memory
  size_t size;/// tranferred size
};
using DMAAddrList = std::list<DMAAddrArea>;

/**
 * @brief Test value if it is 4byte aligned
 * @param val value to be tested
 * @param name the description of value
 *
 * Test a value and throw std::invalid_argument if the value is not
 * a multiple of four. This function is used for validation of
 * DMA source and destination address and size.
 */
void check_4byte_align(uint64_t val, const char *name) {
  if (val % 4 != 0) {
    std::ostringstream msgstr;
    msgstr << name << " (" << std::hex << val << ") is invalid";
    throw std::invalid_argument(msgstr.str());
  }
}

/**
 * @brief push a DMA request to request list
 * @param[in,out] addr_list a list of DMAAddrArea
 * @param gpumem VEHVA of GPU memory to transfer (source or destination)
 * @param vemem VEHVA of VE memory to transfer (source or destination)
 * @param size the size of data transfer
 */
void _pushDMAAddrList(DMAAddrList *addr_list, uint64_t gpumem, uint64_t vemem,
    size_t size) {
  INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_PUSH_DMA_ADDR_START(
        addr_list, gpumem, vemem, size));
  check_4byte_align(gpumem, "GPU VEHVA");
  check_4byte_align(vemem, "VE hmem VEHVA");
  check_4byte_align(size, "Transfer size");
  addr_list->push_back({.vehva_gpumem = gpumem, .vehva_vemem = vemem,
      .size = size});
  INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_PUSH_DMA_ADDR(addr_list,
        gpumem, vemem, size));
}

/**
 * descriptor of GPU memory area mapped from VE for DMA
 *
 * Each MappedGPUMemoryArea specifies a GPU memory area contiguous
 * in VEHVA address space.
 */
struct MappedGPUMemoryArea {
  int device_memory_id;/// device memory ID
  void *vaddr;/// start adddress (in host process address space)
  size_t size;/// the size of area
  uint64_t vehva;/// VEHVA corresponding to vaddr
  size_t offset; // for the convenience of finding the area
};

/**
 * resource for DMA transfer between GPU and VE
 */
class CopyResourceBetweenGPUAndVE {
  CUDAMemoryRegion *gpumemory;/// GPU memory region
  VEOHmemRegion *vememory;/// VE memory region
  std::list<MappedGPUMemoryArea> gpu_area_list;/// list of mapping to GPU memory
  uint64_t vehva_vemem;/// VEHVA at the start of VE memory region

  /**
   * @brief Detach and unregister GPU memory region
   */
  void detachAndUnregisterGPUMemory() {
    auto proc = this->vememory->getProc();
    auto gpu_area_list = this->gpu_area_list;
    for (auto it = gpu_area_list.begin(); it != gpu_area_list.end();
        it = gpu_area_list.erase(it)) {
      int rv;
      rv = veo::wrap::veo_detach_dev_mem(proc, it->vehva);
      INTERDEVCOPY_ASSERT_ZERO(rv);
      rv = veo::wrap::veo_unregister_gpu_mem(proc, it->device_memory_id);
      INTERDEVCOPY_ASSERT_ZERO(rv);
    }
  }

  static constexpr uint64_t VEHVA_UNSET = ~0UL;
  /**
   * @brief unregster mapping to VE memory region
   */
  void unregisterVEMemory() {
    if (this->vehva_vemem == VEHVA_UNSET) {
      // Since VE memory is not registered, do nothing.
      return;
    }
    auto proc = this->vememory->getProc();
    int rv = veo::wrap::veo_unregister_mem_from_dmaatb_unalign(proc,
        this->vehva_vemem);
    INTERDEVCOPY_ASSERT_ZERO(rv);
  }

public:
  /**
   * @brief constructor
   * @param gpumem GPU memory region
   * @param vemem VE memory region
   *
   * Create mapping for VE DMA to GPU and VE memory area.
   */
  CopyResourceBetweenGPUAndVE(CUDAMemoryRegion *gpumem, VEOHmemRegion *vemem):
      gpumemory(gpumem), vememory(vemem), vehva_vemem(VEHVA_UNSET) {
    // refcount of gpumem and vemem are already to be incremented.
    std::exception_ptr ep;
    try {
      // GPU side
      unsigned int flags = 1;
      CUresult curesult = cuda::wrap::cuPointerSetAttribute(&flags,
        CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
        cuda::to_deviceptr(gpumem->getPtr()));
      switch (curesult) {
        case CUDA_SUCCESS:
          break;// OK
        case CUDA_ERROR_INVALID_VALUE:
          throw std::invalid_argument(
              "cuPointerSetAttribute(): CUDA_ERROR_INVALID_VALUE");
        default:
          throw cuda::UnhandleableCUDAError(curesult,
              "cuPointerSetAttribute()");
      }

      // map GPU memory from VE
      auto proc = vemem->getProc();
      void *vaddr = gpumem->getPtr();
      uint64_t remain = gpumem->getSize();
      uint64_t registered_size;
      size_t offset = 0;
      while (1) {
        int device_memory_id = util::errnoToException(
            veo::wrap::veo_register_gpu_mem(proc,
              util::Cast<uint64_t>(vaddr), remain, &registered_size),
            "veo_register_gpu_mem");
        int64_t vehva_gpu = veo::wrap::veo_attach_dev_mem(proc,
            device_memory_id);
        if (vehva_gpu < 0) {
          INTERDEVCOPY_ASSERT_ZERO(veo::wrap::veo_unregister_gpu_mem(proc,
                device_memory_id));
          util::errnoToException(vehva_gpu, "veo_attach_dev_mem");
        }
        this->gpu_area_list.push_back({.device_memory_id = device_memory_id,
            .vaddr = vaddr, .size = registered_size,
            .vehva = util::Cast<uint64_t>(vehva_gpu), .offset = offset});
        INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_REGISTER_GPU_MEM(
              vaddr, registered_size, vehva_gpu, remain));
        vaddr = util::addVoidP(vaddr, registered_size);
        offset += registered_size;
        if( remain <= registered_size) break;
        remain -= registered_size;
      }

      // VE side
      auto vehva_ve = veo::wrap::veo_register_mem_to_dmaatb_unalign(
          vemem->getHmem(), vemem->getSize());
      if (vehva_ve == util::Cast<uint64_t>(-1)) {
        INTERDEVCOPY_ASSERT(errno > 0);
        util::errnoToException(-errno, "veo_register_mem_to_dmaatb_unalign");
      }
      this->vehva_vemem = vehva_ve;
      INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_REGISTER_VE_MEM(
            vemem->getHmem(), vemem->getSize(), vehva_ve));
    } catch (...) {
      ep = std::current_exception();
      // On error, release mapped GPU and VE memory areas to avoid leak.
      this->detachAndUnregisterGPUMemory();
      this->unregisterVEMemory();
    }
    if (ep) {
      std::rethrow_exception(ep);
    }
  }

  /**
   * @brief destructor
   *
   * Free mapping to GPU and VE memory for VE DMA.
   */
  ~CopyResourceBetweenGPUAndVE() {
    this->unregisterVEMemory();
    this->detachAndUnregisterGPUMemory();
  }


  /**
   * @brief get a list of pairs of GPU and VE memory areas
   * @param gpu_off offset of the start of transferred area in GPU memory
   * @param ve_off offset of the start of transferred areain VE memory1:w
   * @param size transfer size in byte.
   * @return a list of DMAAreaAddr to be transferered.
   *
   * Create a list of pairs of contiguous areas of GPU memory and VE memory
   * for DMA transfer. Each area is contiguous but can be larger than the
   * maximum size of transfer that DMA engine supports.
   */
  DMAAddrList getDMAAddrList(size_t gpu_off, size_t ve_off,
      size_t size) const {
    // find the first GPU memory area
    auto gpu_start_area_it = std::find_if(this->gpu_area_list.begin(),
        this->gpu_area_list.end(),
        [gpu_off](const MappedGPUMemoryArea &a)->bool {
          return a.offset <= gpu_off && gpu_off < a.offset + a.size;
        });
    INTERDEVCOPY_ASSERT(gpu_start_area_it != this->gpu_area_list.end());

    DMAAddrList rv;
    size_t gpu_end_offset = gpu_off + size;
    size_t gpu_start_offset_in_page = gpu_off - gpu_start_area_it->offset;
    uint64_t gpu_start_vehva = gpu_start_area_it->vehva
      + gpu_start_offset_in_page;
    uint64_t ve_start_vehva = this->vehva_vemem + ve_off;
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_COPY_FIRST_GPU_AREA(
          this, gpu_off, gpu_start_area_it->vehva, gpu_start_offset_in_page));
    if (gpu_start_offset_in_page + size <= gpu_start_area_it->size) {
      _pushDMAAddrList(&rv, gpu_start_vehva, ve_start_vehva, size);
      return rv;
    } else {
      // The first area does not cover all the range to transfer.
      _pushDMAAddrList(&rv, gpu_start_vehva, ve_start_vehva,
          gpu_start_area_it->size - gpu_start_offset_in_page);
      // continue
    }
    uint64_t vehva_vemem = ve_start_vehva +
      (gpu_start_area_it->size - gpu_start_offset_in_page);
    ++gpu_start_area_it;

    // Add  areas to rv until the last area is found.
    auto last_area_it = std::find_if(gpu_start_area_it,
        this->gpu_area_list.end(),
        [&rv, gpu_end_offset, &vehva_vemem](const MappedGPUMemoryArea &a) {
          if (a.offset + a.size >= gpu_end_offset) {
            // this area is last.
            _pushDMAAddrList(&rv, a.vehva, vehva_vemem,
                gpu_end_offset - a.offset);
            return true;
          } else {
            // Since the area is not last, transfer all.
            _pushDMAAddrList(&rv, a.vehva, vehva_vemem, a.size);
            vehva_vemem += a.size;
            return false;
          }
        });
    INTERDEVCOPY_ASSERT(last_area_it != this->gpu_area_list.end());
    return rv;
  }

  veo_proc_handle *getVEOProc() noexcept {
    return this->vememory->getProc();
  }

  CUDAMemoryRegion *getGPUMemory() noexcept {
    return this->gpumemory;
  }

  VEOHmemRegion *getVEOHmem() noexcept {
    return this->vememory;
  }
};

/// Direction of DMA transfer (for template arguments)
enum class DMADirection {
  FromGPUToVE,
  FromVEToGPU,
};

/**
 * @brief post VE DMA request between GPU and VE
 * @tparam dir direction of DMA transfer
 * @param proc VEO process handle to handle a DMA request
 * @param gpumem VEHVA of GPU memory
 * @param vemem VEHVA of VE memory
 * @param size the size of data transfer
 * @param dmahdl[out] DMA handle for check DMA request status
 * @return zero upon success; negative upon failure.
 *         See veo_dma_post() in VEO API reference.
 *
 * Post a DMA request between GPU and VE.
 * This template function reorders source and destination arguments of VEO
 * VE DMA API to implement transfer both from GPU to VE and from VE to GPU
 * as a common template code.
 */
template <DMADirection dir> int postDMA(veo_proc_handle *proc, uint64_t gpumem,
    uint64_t vemem, size_t size, ve_dma_handle_t *dmahdl);

template<> int postDMA<DMADirection::FromVEToGPU>(veo_proc_handle *proc,
    uint64_t gpumem, uint64_t vemem, size_t size, ve_dma_handle_t *dmahdl) {
  return veo::wrap::veo_dma_post(proc, gpumem, vemem, size, dmahdl);
}

template <> int postDMA<DMADirection::FromGPUToVE>(veo_proc_handle *proc,
  uint64_t gpumem, uint64_t vemem, size_t size, ve_dma_handle_t *dmahdl) {
  return veo::wrap::veo_dma_post(proc, vemem, gpumem, size, dmahdl);
}

/**
 * DMA request descriptor to hold a pair of DMA handle and the transfer size
 */
struct DMARequest {
  ve_dma_handle_t handle;
  size_t size;
};
using DMARequestList = std::list<DMARequest>;

/**
 * @brief divide and post DMA transfer requests
 * @tparam dir direction of DMA transfer
 * @param proc VEO process handle to handle a DMA request
 * @param[in,out] dma_addr_list a list of GPU and VE memory area pairs
 * @param[in,out] request_list a list of DMA request descriptors
 * @retval 0 the DMA request completes.
 * @retval -EAGAIN some requests are left to be posted yet.
 *
 * The function tryToPost()
 *  - divides each area in dma_addr_list into one or more areas in the length
 *   which DMA engine can transfer,
 *  - post DMA requests to transfer the areas, and
 *  - append the requests to request_list.
 */
template <DMADirection dir> int tryToPost(veo_proc_handle *proc,
    DMAAddrList &dma_addr_list, DMARequestList &request_list) {
  INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_TRY_TO_POST_START(dir,
        dma_addr_list.size()));
  for (auto it = dma_addr_list.begin(); it != dma_addr_list.end();
      it = dma_addr_list.erase(it)) {
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_POP_REQUEST_AREA(dir));
    while (it->size > 0) {
      size_t reqsize = std::min(it->size, MAX_DMA_REQUEST_SIZE);
      DMARequest req = {{}, reqsize};
      int rv = postDMA<dir>(proc, it->vehva_gpumem, it->vehva_vemem,
          reqsize, &req.handle);
      INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_POST_DMA_AREA(dir,
            it->vehva_gpumem, it->vehva_vemem, reqsize, rv,
            req.handle.index));
      INTERDEVCOPY_ASSERT(rv == 0 || rv == -EAGAIN);
      if (rv == -EAGAIN) {
        return -EAGAIN;
      }
      request_list.push_back(req);
      it->vehva_gpumem += reqsize;
      it->vehva_vemem += reqsize;
      it->size -= reqsize;
    }
  }
  INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_TRY_TO_POST_COMPLETE(dir));
  return 0;
}

/**
 * @brief try to complete DMA requests
 * @param proc VEO process handle to handle a DMA request
 * @param[in,out] request_list a list of DMA request descriptors
 * @param[in,out] transferred the size of data transferred.
 * @retval true all DMA requests in request_list complete.
 * @retval false one or more DMA requests remain yet.
 *
 * Test the status of DMA requests, remove completed request
 * from request_list and add the size of data transfer to transferred.
 */
bool tryToCompleteRequests(veo_proc_handle *proc,
    DMARequestList &request_list, ssize_t &transferred) {
  if (INTERDEVCOPY_TRACE_ENABLED(
        LIBINTERDEVCOPY_CHANNEL_GPUVE_TRY_TO_COMPLETE_START)) {
    // avoid list.size() call unless the probe is enabled.
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_TRY_TO_COMPLETE_START(
        request_list.size(), transferred));
  }
  if (request_list.empty()) {
    return true;
  }
  {
    auto result = request_list.end();
    util::Finally f([&request_list, &result]() {
      request_list.erase(result, request_list.end());
    });
    result = std::remove_if(request_list.begin(), request_list.end(),
        [proc, &transferred] (DMARequest &req) {
      int ret = veo::wrap::veo_dma_poll(proc, &req.handle);
      INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_POLL_DMA_RESULT(
          req.handle.index, ret));
      if (ret == 0) {
        transferred += req.size;
        return true;
      } else if (ret == -EAGAIN) {
        return false;
      } else if (ret < 0) {
        util::_errnoToException(ret, "veo_dma_poll()");
      } else {
        // DMA failed unexpectedly
        INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_DMA_EXCEPTION(
            req.handle.index, ret));
        throw veo::UnhandleableVEOError(ret, "veo_dma_poll()");
      }
    });
  }
  if (INTERDEVCOPY_TRACE_ENABLED(
        LIBINTERDEVCOPY_CHANNEL_GPUVE_TRY_TO_COMPLETE_RETURN)) {
    INTERDEVCOPY_TRACE(LIBINTERDEVCOPY_CHANNEL_GPUVE_TRY_TO_COMPLETE_RETURN(
          request_list.size(), transferred));
  }
  return request_list.empty();
}

/**
 * @brief perform DMA transfer
 * @tparam dir direction of DMA transfer
 * @param proc VEO process handle to handle a DMA request
 * @param dma_addr_list list of GPU and VE memory area pairs to transfer
 * @return size of transferred data.
 *
 * Post all DMA requests and poll until their completion.
 * The request list dma_addr_list is moved and not supposed to be used
 * after calling this function to avoid copy.
 */
template <DMADirection dir> ssize_t performDMA(veo_proc_handle *proc,
    DMAAddrList &&dma_addr_list) {
  DMARequestList requests;
  ssize_t transferred = 0;
  bool complete = false;
  int rv;
  do {
    rv = tryToPost<dir>(proc, dma_addr_list, requests);
    complete = tryToCompleteRequests(proc, requests, transferred);
  } while (!dma_addr_list.empty() && rv == -EAGAIN);
  // Since an exception is to be thrown on errors, all requests are
  // expected to be posted successfully here.
  INTERDEVCOPY_ASSERT_ZERO(rv);
  while (!complete) {
    complete = tryToCompleteRequests(proc, requests, transferred);
  }
  return transferred;
}

/**
 * @brief copy function
 */
template <DMADirection dir>
ssize_t doDirectCopy(CopyResourceBetweenGPUAndVE *resource,
    size_t gpuoff, size_t veoff, size_t size) {
  auto dma_addr_list = resource->getDMAAddrList(gpuoff, veoff, size);
  return performDMA<dir>(resource->getVEOProc(), std::move(dma_addr_list));
}

/**
 * CopyFuncGetter for transfer from GPU to VE memory
 */
struct CopyFuncGetterFromGPUToVE {
  using srctype = CUDAMemoryRegion;
  using desttype = VEOHmemRegion;
  // Getter operator implementation
  CopyFuncType operator()(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
      __attribute__((unused)) void *option) {
    auto gpumem = dynamic_cast<CUDAMemoryRegion *>(src);
    INTERDEVCOPY_ASSERT(gpumem != nullptr);
    auto vemem = dynamic_cast<VEOHmemRegion *>(dst);
    INTERDEVCOPY_ASSERT(vemem != nullptr);
    // Create resources for DMA between GPU and VE.
    // unique_ptr does not work because lambda object is copied on return.
    auto res = std::make_shared<CopyResourceBetweenGPUAndVE>(gpumem, vemem);
    return [r = std::move(res)](size_t dstoff, size_t srcoff, size_t size,
        __attribute__((unused)) void *opt) {
      return doDirectCopy<DMADirection::FromGPUToVE>(r.get(),
          srcoff, dstoff, size);
    };
  }
};

/**
 * CopyFuncGetter for transfer from VE to GPU memory
 */
struct CopyFuncGetterFromVEToGPU {
  using srctype = VEOHmemRegion;
  using desttype = CUDAMemoryRegion;
  // Getter operator implementation
  CopyFuncType operator()(DeviceMemoryRegion *dst, DeviceMemoryRegion *src,
      __attribute__((unused)) void *option) {
    auto gpumem = dynamic_cast<CUDAMemoryRegion *>(dst);
    INTERDEVCOPY_ASSERT(gpumem != nullptr);
    auto vemem = dynamic_cast<VEOHmemRegion *>(src);
    INTERDEVCOPY_ASSERT(vemem != nullptr);
    // Create resources for DMA between GPU and VE.
    // unique_ptr does not work because lambda object is copied on return.
    auto res = std::make_shared<CopyResourceBetweenGPUAndVE>(gpumem, vemem);
    return [r = std::move(res)](size_t dstoff,
        size_t srcoff, size_t size, __attribute__((unused)) void *opt) {

      uint32_t word;
      INTERDEVCOPY_ASSERT_ZERO(veo::wrap::veo_hmemcpy(&word,
            r->getVEOHmem()->getHmem(srcoff), sizeof(word)));
      auto rv = doDirectCopy<DMADirection::FromVEToGPU>(r.get(),
          dstoff, srcoff, size);
      // confirm the completion of write by ordering rule.
      INTERDEVCOPY_ASSERT(cuda::wrap::cudaMemcpy(
            r->getGPUMemory()->getPtr(dstoff), &word, sizeof(word),
            cudaMemcpyHostToDevice) == cudaSuccess);
      return rv;
    };
  }
};

} // namespace interdevcopy::beween_gpu_ve
} // namespace interdevcopy

REGISTER_COPY_HANDLER_IF(
    interdevcopy::between_gpu_ve::CopyFuncGetterFromGPUToVE,
    interdevcopy::cuda::init_wrapper() && interdevcopy::veo::init_wrapper());
REGISTER_COPY_HANDLER_IF(
    interdevcopy::between_gpu_ve::CopyFuncGetterFromVEToGPU,
    interdevcopy::cuda::init_wrapper() && interdevcopy::veo::init_wrapper());
