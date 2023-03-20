/**
 * @example ve-host_copy.c
 */
#include <stdlib.h>
#include <stdio.h>
#include <ve_offload.h>
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

  // alloc resources
  void *vesrc;
  void *vedst;
  ret = veo_alloc_hmem(proc, &vesrc, size);
  if (ret != 0) return err("veo_alloc_hmem", ret);
  ret = veo_alloc_hmem(proc, &vedst, size);
  if (ret != 0) return err("veo_alloc_hmem", ret);
  
  char* hostsrc = malloc(sizeof(char) * size);
  char* hostdst = malloc(sizeof(char) * size);

  // VE -> Host
  struct interdevcopy_memory_region *vesrc_mr = interdevcopy_create_memory_region(
    vesrc, size, INTERDEVCOPY_MEMORY_VE_HMEM, NULL);
  if (vesrc_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)vesrc_mr);
  struct interdevcopy_memory_region *hostdst_mr = interdevcopy_create_memory_region(
    hostdst, size, INTERDEVCOPY_MEMORY_HOST_MEM, NULL);
  if (hostdst_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)hostdst_mr);
  struct interdevcopy_channel *ve_host_ch = interdevcopy_create_channel(
    hostdst_mr, vesrc_mr, NULL);
  if (ve_host_ch < 0) return err("interdevcopy_create_channel", (intptr_t)ve_host_ch);
  ssize_t ve_host_val = interdevcopy_copy(ve_host_ch, 0, 0, size, NULL);
  if (ve_host_val != size) return err("interdevcopy_copy", (int)ve_host_val);
  ret = interdevcopy_destroy_channel(ve_host_ch);
  if (ret != 0) return err("interdevcopy_destroy_channel", ret);
  ret = interdevcopy_destroy_memory_region(vesrc_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);
  ret = interdevcopy_destroy_memory_region(hostdst_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);

  // Host -> VE
  struct interdevcopy_memory_region *hostsrc_mr = interdevcopy_create_memory_region(
    hostsrc, size, INTERDEVCOPY_MEMORY_HOST_MEM, NULL);
  if (hostsrc_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)hostsrc_mr);
  struct interdevcopy_memory_region *vedst_mr = interdevcopy_create_memory_region(
    vedst, size, INTERDEVCOPY_MEMORY_VE_HMEM, NULL);
  if (vedst_mr < 0) return err("interdevcopy_create_memory_region", (intptr_t)vedst_mr);
  struct interdevcopy_channel *host_ve_ch = interdevcopy_create_channel(
    vedst_mr, hostsrc_mr, NULL);
  if (host_ve_ch < 0) return err("interdevcopy_create_channel", (intptr_t)host_ve_ch);
  ssize_t host_ve_val = interdevcopy_copy(host_ve_ch, 0, 0, size, NULL);
  if (host_ve_val != size) return err("interdevcopy_copy", (int)host_ve_val);
  ret = interdevcopy_destroy_channel(host_ve_ch);
  if (ret != 0) return err("interdevcopy_destroy_channel", ret);
  ret = interdevcopy_destroy_memory_region(hostsrc_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);
  ret = interdevcopy_destroy_memory_region(vedst_mr);
  if (ret != 0) return err("interdevcopy_destroy_memory_region", ret);

  // free resorces
  free(hostsrc);
  free(hostdst);
  ret = veo_free_hmem(vesrc);
  if (ret != 0) return err("veo_free_hmem", ret);
  ret = veo_free_hmem(vedst);
  if (ret != 0) return err("veo_free_hmem", ret);

  ret = veo_proc_destroy(proc);
  if (ret != 0) return err("veo_proc_destroy", ret);

  return 0;
}
