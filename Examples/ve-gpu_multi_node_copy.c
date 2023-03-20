/**
 * @example ve-gpu_multi_node_copy.c
 */
#include <stdlib.h>
#include <stdio.h>
#include <ve_offload.h>
#include <cuda_runtime.h>
#include <veosinfo/veosinfo.h>
#include <interdevcopy.h>

int err(char* msg, int ret){
  printf("%s failed ret = %d\n", msg, ret);
  return -1;
}

int main(int argc, char **argv) {
  ssize_t size = 1 << 22;
  int ret = 0;

  // ve node setup
  int ve_node_count = 0;
  struct ve_nodeinfo node_info;
  if(-1 == ve_node_info(&node_info)) return err("ve_node_info", -1);
  ve_node_count = node_info.total_node_count;

  struct veo_proc_handle** proc_list = malloc(
    ve_node_count * sizeof(struct veo_proc_handle*));
  for(int i = 0; i < ve_node_count; i++ ) {
    proc_list[i] = NULL;
  }

  for(int i = 0; i < ve_node_count; i++ ) {
    int status = ve_check_node_status(node_info.nodeid[i]);
    if( 0 != status ) continue;

    struct veo_proc_handle* tmp_proc = veo_proc_create(node_info.nodeid[i]);
    if( tmp_proc == NULL ) continue;
    proc_list[i] = tmp_proc;
  }

  // cuda setup
  int cuda_node_count = 0;
  ret = cudaGetDeviceCount(&cuda_node_count);
  if (ret != cudaSuccess) return err("cudaGetDeviceCount", ret);
 
  for(int i=0; i < ve_node_count; i++){
    if( proc_list[i] == NULL ) continue;

    for(int j=0; j < cuda_node_count; j++){
      // alloc resource
      void *vesrc;
      void *vedst;
      void *gpusrc;
      void *gpudst;
      ret = cudaSetDevice(j);
      if (ret != cudaSuccess) err("cudaSetDevice", ret);
      ret = veo_alloc_hmem(proc_list[i], &vesrc, size);
      if (ret != 0) return err("veo_alloc_hmem", ret);
      ret = veo_alloc_hmem(proc_list[i], &vedst, size);
      if (ret != 0) return err("veo_alloc_hmem", ret);
      ret = cudaMalloc(&gpusrc, size);
      if (ret != cudaSuccess) return err("cudaMalloc", ret);
      ret = cudaMalloc(&gpudst, size);
      if (ret != cudaSuccess) return err("cudaMalloc", ret);

      // VE(i) -> GPU(j)
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

      // GPU(j) -> VE(i)
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
    }
  }

  for(int i=0; i < ve_node_count; i++){
    if (proc_list[i] != NULL) {
      ret = veo_proc_destroy(proc_list[i]);
      if (ret != 0) return err("veo_proc_destroy", ret);
    }
  }
  free(proc_list);

  return 0;
}
