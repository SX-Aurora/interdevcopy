/**
 * @example ve-gpu_copy.c
 */
#include <stdlib.h>
#include <stdio.h>
#include <ve_offload.h>
#include <cuda_runtime.h>
#include <interdevcopy.h>

int err(char* msg, int ret){
  printf("%s failed ret = %d\n", msg, ret);
  return -1;
}

int main(int argc, char **argv) {
  ssize_t size = 1 << 22;
  int ret = 0;

  // ve node setup
  struct veo_proc_handle* proc = veo_proc_create(-1);
  if (proc == NULL) return err("veo_proc_create(-1)", -1);

  // alloc resource
  void *vesrc;
  void *vedst;
  void *gpusrc;
  void *gpudst;
  ret = veo_alloc_hmem(proc, &vesrc, size);
  if (ret != 0) return err("veo_alloc_hmem", ret);
  ret = veo_alloc_hmem(proc, &vedst, size);
  if (ret != 0) return err("veo_alloc_hmem", ret);
  ret = cudaMalloc(&gpusrc, size);
  if (ret != cudaSuccess) return err("cudaMalloc", ret);
  ret = cudaMalloc(&gpudst, size);
  if (ret != cudaSuccess) return err("cudaMalloc", ret);

  // VE -> GPU
  struct interdevcopy_memory_region *vesrc_mr = interdevcopy_create_memory_region(
    vesrc, size, INTERDEVCOPY_MEMORY_VE_HMEM, NULL);
  if (vesrc_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)vesrc_mr);
  struct interdevcopy_memory_region *gpudst_mr = interdevcopy_create_memory_region(
    gpudst, size, INTERDEVCOPY_MEMORY_CUDA_MEM, NULL);
  if (gpudst_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)gpudst_mr);
  struct interdevcopy_channel *ve_gpu_ch = interdevcopy_create_channel(
    gpudst_mr, vesrc_mr, NULL);
  if (ve_gpu_ch < 0) return err("interdevcopy_create_channel", (intptr_t)ve_gpu_ch);
  ssize_t ve_gpu_val = interdevcopy_copy(ve_gpu_ch, 0, 0, size, NULL);
  if (ve_gpu_val != size) return err("interdevcopy_copy", (int)ve_gpu_val);
  ret = interdevcopy_destroy_channel(ve_gpu_ch);
  if (ret != 0) return err("interdevcopy_destroy_channel", ret);
  ret = interdevcopy_destroy_memory_region(vesrc_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);
  ret = interdevcopy_destroy_memory_region(gpudst_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);

  // GPU -> VE
  struct interdevcopy_memory_region *gpusrc_mr = interdevcopy_create_memory_region(
    gpusrc, size, INTERDEVCOPY_MEMORY_CUDA_MEM, NULL);
  if (gpusrc_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)gpusrc_mr);
  struct interdevcopy_memory_region *vedst_mr = interdevcopy_create_memory_region(
    vedst, size, INTERDEVCOPY_MEMORY_VE_HMEM, NULL);
  if (vedst_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)vedst_mr);
  struct interdevcopy_channel *gpu_ve_ch = interdevcopy_create_channel(
    vedst_mr, gpusrc_mr, NULL);
  if (gpu_ve_ch < 0) return err("interdevcopy_create_channel", (intptr_t)gpu_ve_ch);
  ssize_t gpu_ve_val = interdevcopy_copy(gpu_ve_ch, 0, 0, size, NULL);
  if (gpu_ve_val != size) return err("interdevcopy_copy", (int)gpu_ve_val);
  ret = interdevcopy_destroy_channel(gpu_ve_ch);
  if (ret != 0) return err("interdevcopy_destroy_channel", ret);
  ret = interdevcopy_destroy_memory_region(gpusrc_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);
  ret = interdevcopy_destroy_memory_region(vedst_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);

  // free resorce
  ret = cudaFree(gpudst);
  if (ret != cudaSuccess) return err("cudaFree", ret);
  ret = cudaFree(gpusrc);
  if (ret != cudaSuccess) return err("cudaFree", ret);
  ret = veo_free_hmem(vesrc);
  if (ret != 0) return err("veo_free_hmem", ret);
  ret = veo_free_hmem(vedst);
  if (ret != 0) return err("veo_free_hmem", ret);
  ret = veo_proc_destroy(proc);
  if (ret != 0) return err("veo_proc_destroy", ret);

  return 0;
}
