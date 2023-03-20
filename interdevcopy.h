/*
 * Inter-device copy library
 *
 * Copyright (C) 2023 NEC Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * @file interdevcopy.h
 * @brief Interdevcopy library API header
 */

#ifndef INTERDEVCOPY_H_INCLUDE_
#define INTERDEVCOPY_H_INCLUDE_

#define INTERDEVCOPY_API_VERSION (1)

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct interdevcopy_memory_region;
struct interdevcopy_channel;

/**
 * @ingroup API
 * @brief device memory type
 */
enum interdevcopy_memory_type {
  INTERDEVCOPY_MEMORY_HOST_MEM = 0,
  INTERDEVCOPY_MEMORY_VE_HMEM,
  INTERDEVCOPY_MEMORY_CUDA_MEM,
  INTERDEVCOPY_MEMORY_TYPE_MAX,
};
/**
 * @var interdevcopy_memory_type::INTERDEVCOPY_MEMORY_HOST_MEM
 * @brief Buffer is on host memory case
 * @var interdevcopy_memory_type::INTERDEVCOPY_MEMORY_VE_HMEM
 * @brief Buffer is on VE memory case
 * @var interdevcopy_memory_type::INTERDEVCOPY_MEMORY_CUDA_MEM
 * @brief Buffer is on CUDAGPU memory case
 * @var interdevcopy_memory_type::INTERDEVCOPY_MEMORY_TYPE_MAX
 * @brief Default value
 */

struct interdevcopy_memory_region *interdevcopy_create_memory_region(void *,
    size_t, enum interdevcopy_memory_type, void *);
int interdevcopy_destroy_memory_region(struct interdevcopy_memory_region *);

struct interdevcopy_channel *interdevcopy_create_channel(
    struct interdevcopy_memory_region *, struct interdevcopy_memory_region *,
    void *);
int interdevcopy_destroy_channel(struct interdevcopy_channel *);

ssize_t interdevcopy_copy(struct interdevcopy_channel *,
    unsigned long, unsigned long, size_t, void *);

#ifdef __cplusplus
} // extern "C"
#endif
#endif
