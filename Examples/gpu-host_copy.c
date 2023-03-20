/**
 * @example gpu-host_copy.c
 */
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <interdevcopy.h>

int err(char* msg, int ret){
  printf("%s failed ret = %d\n", msg, ret);
  return -1;
}

int main(int argc, char **argv) {
  ssize_t size = 1 << 22;
  int ret = 0;

  // alloc resource
  void *gpusrc;
  void *gpudst;
  ret = cudaMalloc(&gpusrc, size);
  if (ret != cudaSuccess) return err("cudaMalloc", ret);
  ret = cudaMalloc(&gpudst, size);
  if (ret != cudaSuccess) return err("cudaMalloc", ret);

  char* hostsrc = malloc(sizeof(char) * size);
  char* hostdst = malloc(sizeof(char) * size);

  // GPU -> Host
  struct interdevcopy_memory_region *gpusrc_mr = interdevcopy_create_memory_region(
    gpusrc, size, INTERDEVCOPY_MEMORY_CUDA_MEM, NULL);
  if (gpusrc_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)gpusrc_mr);
  struct interdevcopy_memory_region *hostdst_mr = interdevcopy_create_memory_region(
    hostdst, size, INTERDEVCOPY_MEMORY_HOST_MEM, NULL);
  if (hostdst_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)hostdst_mr);
  struct interdevcopy_channel *gpu_host_ch = interdevcopy_create_channel(
    hostdst_mr, gpusrc_mr, NULL);
  if (gpu_host_ch < 0) return err("interdevcopy_create_channel", (intptr_t)gpu_host_ch);
  ssize_t gpu_host_val = interdevcopy_copy(gpu_host_ch, 0, 0, size, NULL);
  if (gpu_host_val != size) return err("interdevcopy_copy", (int)gpu_host_val);
  ret = interdevcopy_destroy_channel(gpu_host_ch);
  if (ret != 0) return err("interdevcopy_destroy_channel", ret);
  ret = interdevcopy_destroy_memory_region(gpusrc_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);
  ret = interdevcopy_destroy_memory_region(hostdst_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);

  // Host -> GPU
  struct interdevcopy_memory_region *hostsrc_mr = interdevcopy_create_memory_region(
    hostsrc, size, INTERDEVCOPY_MEMORY_HOST_MEM, NULL);
  if (hostsrc_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)hostsrc_mr);
  struct interdevcopy_memory_region *gpudst_mr = interdevcopy_create_memory_region(
    gpudst, size, INTERDEVCOPY_MEMORY_CUDA_MEM, NULL);
  if (gpudst_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)gpudst_mr);
  struct interdevcopy_channel *host_gpu_ch = interdevcopy_create_channel(
    gpudst_mr, hostsrc_mr, NULL);
  if (host_gpu_ch < 0) return err("interdevcopy_create_channel", (intptr_t)host_gpu_ch);
  ssize_t host_gpu_val = interdevcopy_copy(host_gpu_ch, 0, 0, size, NULL);
  if (host_gpu_val != size) return err("interdevcopy_copy", (int)host_gpu_val);
  ret = interdevcopy_destroy_channel(host_gpu_ch);
  if (ret != 0) return err("interdevcopy_destroy_channel", ret);
  ret = interdevcopy_destroy_memory_region(hostsrc_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);
  ret = interdevcopy_destroy_memory_region(gpudst_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);

  // free resorce
  ret = cudaFree(gpudst);
  if (ret != cudaSuccess) return err("cudaFree", ret);
  ret = cudaFree(gpusrc);
  if (ret != cudaSuccess) return err("cudaFree", ret);
  free(hostdst);
  free(hostsrc);

  return 0;
}
