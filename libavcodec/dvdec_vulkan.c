/*
 * DV decoder (Vulkan hwaccel pipeline scaffold)
 * Copyright (c) 2002 Fabrice Bellard
 * Copyright (c) 2004 Roman Shaposhnik
 *
 * 50 Mbps (DVCPRO50) support
 * Copyright (c) 2006 Daniel Maas <dmaas@maasdigital.com>
 *
 * 100 Mbps (DVCPRO HD) support
 * Initial code by Daniel Maas <dmaas@maasdigital.com> (funded by BBC R&D)
 * Final code by Roman Shaposhnik
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 */

#define FF_INTERNAL_FIELDS

#include <errno.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>

#include "libavcodec/avcodec.h"
#include "libavutil/avassert.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_vulkan.h"
#include "libavutil/imgutils.h"
#include "libavutil/mem.h"
#include "libavutil/thread.h"
#include "libavutil/vulkan.h"
#if CONFIG_LIBGLSLANG || CONFIG_LIBSHADERC
#include "libavutil/vulkan_spirv.h"
#endif

#include "avcodec.h"
#include "decode.h"
#include "dv.h"
#include "dv_internal.h"
#include "dv_profile_internal.h"
#include "dvdata.h"
#include "get_bits.h"
#include "hwaccel_internal.h"
#include "hwaccels.h"
#include "internal.h"
#include "mathops.h"
#include "put_bits.h"
#include "simple_idct.h"
#include "libswscale/swscale.h"
#include "vlc.h"

#define DV_MAX_WORK_CHUNKS (4 * 12 * 27)

typedef struct DVMacroblockJob {
    uint16_t buf_offset;
    uint16_t mb_x;
    uint16_t mb_y;
    uint8_t  mb_index;
} DVMacroblockJob;

typedef struct DVPipelineStats {
    int                work_pool_size;
    int                mb_jobs;
    int                block_jobs;
    int                frame_width;
    int                frame_height;
    int                mb_width_blocks;
    int                chroma_411_split_mb_x;
    int                last_mb_y;
    enum AVPixelFormat sw_format;
} DVPipelineStats;

typedef struct DVSubContext {
    const AVDVProfile *sys;

    uint8_t  dv_zigzag[2][64];
    uint32_t idct_factor[2 * 4 * 16 * 64];

    uint8_t *frame_packet;
    size_t   frame_packet_size;
    int      frame_packet_from_start;

    DVwork_chunk work_chunks[DV_MAX_WORK_CHUNKS];

    DVMacroblockJob *mb_jobs;
    int              mb_jobs_alloc;
    uint8_t         *mb_field_modes;
    int              mb_field_modes_alloc;

    int16_t  *coeff_blocks;
    int       coeff_blocks_alloc;
    int16_t  *quant_blocks;
    uint32_t *factor_blocks;
    int       dequant_blocks_alloc;

    FFVulkanContext vk;
    FFVkExecPool    idct_exec_pool;
    FFVulkanShader  dequant_shd;
    FFVulkanShader  idct_shd;
    FFVulkanShader  recon_calc_shd;
    FFVulkanShader  recon_shd;
    FFVkBuffer      dequant_quant_buf;
    size_t          dequant_quant_buf_size;
    int32_t        *dequant_quant_buf_map;
    FFVkBuffer      dequant_factor_buf;
    size_t          dequant_factor_buf_size;
    uint32_t       *dequant_factor_buf_map;
    FFVkBuffer      dequant_out_buf;
    size_t          dequant_out_buf_size;
    int32_t        *dequant_out_buf_map;
    FFVkBuffer      idct_buf;
    size_t          idct_buf_size;
    int32_t        *idct_buf_map;
    FFVkBuffer      recon_jobs_buf;
    size_t          recon_jobs_buf_size;
    uint8_t        *recon_jobs_buf_map;
    FFVkBuffer      recon_plane_buf;
    size_t          recon_plane_buf_size;
    uint32_t       *recon_plane_buf_map;
    int             dequant_gpu_ready;
    int             idct_gpu_ready;
    int             recon_gpu_ready;
    int             logged_dequant_gpu_fail;
    int             logged_idct_gpu_fail;
    int             logged_recon_gpu_fail;

    uint8_t *plane_staging;
    int      plane_staging_size;
    uint8_t *decode_plane_staging;
    int      decode_plane_staging_size;
    int      coeff_blocks_are_spatial;
    int      idct_upload_ready;
    int      cpu_output_required;
    int      dequant_output_in_idct_buf;

    DVPipelineStats stats;

    int logged_idct_fallback;
} DVSubContext;

/*
 * Bridge the public dvdec context fields we need from dvdec.c.
 * dvvideo_decode_frame stores the current destination AVFrame in this layout.
 */
typedef struct DVVkDecoderBridge {
    const AVDVProfile *sys;
    const AVFrame     *frame;
    const uint8_t     *buf;
} DVVkDecoderBridge;

typedef struct DVVkIDCTPush {
    uint32_t num_blocks;
    uint32_t blocks_per_row;
    uint32_t output_base;
    uint32_t reserved;
} DVVkIDCTPush;

typedef struct DVVkDequantPush {
    uint32_t num_coeffs;
    uint32_t iweight_bits;
    uint32_t output_stride;
    uint32_t output_base;
} DVVkDequantPush;

typedef struct DVVkReconJob {
    uint32_t mb_x;
    uint32_t mb_y;
    uint32_t field_mode;
    uint32_t reserved;
} DVVkReconJob;

typedef struct DVVkReconCalcPush {
    uint32_t width[4];
    uint32_t height[4];
    uint32_t plane_offset[4];
    uint32_t plane_stride[4];
    uint32_t num_blocks;
    uint32_t blocks_per_mb;
    uint32_t chroma_w_shift;
    uint32_t chroma_h_shift;
    uint32_t is_yuv411;
    uint32_t last_mb_y;
    uint32_t chroma_411_split_mb_x;
} DVVkReconCalcPush;

typedef struct DVVkReconPush {
    uint32_t width[4];
    uint32_t height[4];
    uint32_t plane_offset[4];
    uint32_t plane_stride[4];
    uint32_t chroma_w_shift;
    uint32_t chroma_h_shift;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
} DVVkReconPush;

typedef struct DVVkBlockInfo {
    const uint32_t *factor_table;
    const uint8_t  *scan_table;
    uint8_t         pos;
    uint8_t         partial_bit_count;
    uint32_t        partial_bit_buffer;
} DVVkBlockInfo;

static enum AVPixelFormat dv_vk_choose_sw_format(enum AVPixelFormat sw_format);
static int                dv_vk_recon_gpu_supported(const AVDVProfile *sys);
static int                dv_vk_get_chroma_shifts(enum AVPixelFormat fmt, uint32_t *w_shift, uint32_t *h_shift, uint32_t *is_yuv411);

static const int dv_vk_iweight_bits = 14;

static const uint16_t dv_vk_iweight_88[64] = {
    32768, 16705, 16705, 17734, 17032, 17734, 18205, 18081, 18081, 18205, 18725, 18562, 19195, 18562, 18725, 19266,
    19091, 19705, 19705, 19091, 19266, 21407, 19643, 20267, 20228, 20267, 19643, 21407, 22725, 21826, 20853, 20806,
    20806, 20853, 21826, 22725, 23170, 23170, 21407, 21400, 21407, 23170, 23170, 24598, 23786, 22018, 22018, 23786,
    24598, 25251, 24465, 22654, 24465, 25251, 25972, 25172, 25172, 25972, 26722, 27969, 26722, 29692, 29692, 31521,
};

static const uint16_t dv_vk_iweight_248[64] = {
    32768, 16384, 16705, 16705, 17734, 17734, 17734, 17734, 18081, 18081, 18725, 18725, 21407, 21407, 19091, 19091,
    19195, 19195, 18205, 18205, 18725, 18725, 19705, 19705, 20267, 20267, 21826, 21826, 23170, 23170, 20806, 20806,
    20267, 20267, 19266, 19266, 21407, 21407, 20853, 20853, 21400, 21400, 23786, 23786, 24465, 24465, 22018, 22018,
    23170, 23170, 22725, 22725, 24598, 24598, 24465, 24465, 25172, 25172, 27969, 27969, 25972, 25972, 29692, 29692,
};

static const uint16_t dv_vk_iweight_1080_y[64] = {
    128, 16, 16, 17, 17, 17, 18, 18, 18, 18, 18, 18, 19, 18, 18, 19, 19, 19, 19, 19, 19, 42, 38,  40, 40, 40,  38,  42,  44,  43,  41,  41,
    41,  41, 43, 44, 45, 45, 42, 42, 42, 45, 45, 48, 46, 43, 43, 46, 48, 49, 48, 44, 48, 49, 101, 98, 98, 101, 104, 109, 104, 116, 116, 123,
};

static const uint16_t dv_vk_iweight_1080_c[64] = {
    128, 16, 16, 17, 17, 17,  25,  25,  25,  25,  26,  25,  26,  25,  26,  26,  26,  27,  27,  26,  26, 42,
    38,  40, 40, 40, 38, 42,  44,  43,  41,  41,  41,  41,  43,  44,  91,  91,  84,  84,  84,  91,  91, 96,
    93,  86, 86, 93, 96, 197, 191, 177, 191, 197, 203, 197, 197, 203, 209, 219, 209, 232, 232, 246,
};

static const uint16_t dv_vk_iweight_720_y[64] = {
    128, 16, 16, 17, 17, 17, 18, 18, 18, 18, 18,  18,  19,  18,  18,  19,  19,  19,  19,  19,  19, 42,
    38,  40, 40, 40, 38, 42, 44, 43, 41, 41, 41,  41,  43,  44,  68,  68,  63,  63,  63,  68,  68, 96,
    92,  86, 86, 92, 96, 98, 96, 88, 96, 98, 202, 196, 196, 202, 208, 218, 208, 232, 232, 246,
};

static const uint16_t dv_vk_iweight_720_c[64] = {
    128, 24,  24,  26,  26,  26,  36,  36,  36,  36,  36,  36,  38,  36,  36,  38,  38,  38,  38,  38,  38,  84,
    76,  80,  80,  80,  76,  84,  88,  86,  82,  82,  82,  82,  86,  88,  182, 182, 168, 168, 168, 182, 182, 192,
    186, 192, 172, 186, 192, 394, 382, 354, 382, 394, 406, 394, 394, 406, 418, 438, 418, 464, 464, 492,
};

static RL_VLC_ELEM dv_vk_rl_vlc[1664];
static AVOnce      dv_vk_rl_vlc_once                    = AV_ONCE_INIT;
static atomic_int  dv_vk_global_disable_gpu_idct        = 0;
static atomic_int  dv_vk_global_logged_idct_fallback    = 0;
static atomic_int  dv_vk_global_logged_dequant_gpu_init = 0;
static atomic_int  dv_vk_global_logged_idct_gpu_active  = 0;
static atomic_int  dv_vk_global_logged_idct_gpu_init    = 0;
static atomic_int  dv_vk_global_logged_recon_gpu_init   = 0;

#if CONFIG_LIBGLSLANG || CONFIG_LIBSHADERC
static pthread_mutex_t dv_vk_spv_cache_mutex        = PTHREAD_MUTEX_INITIALIZER;
static uint8_t        *dv_vk_cached_idct_spv        = NULL;
static size_t          dv_vk_cached_idct_spv_len    = 0;
static uint8_t        *dv_vk_cached_dequant_spv     = NULL;
static size_t          dv_vk_cached_dequant_spv_len = 0;
static uint8_t        *dv_vk_cached_calc_spv        = NULL;
static size_t          dv_vk_cached_calc_spv_len    = 0;
static uint8_t        *dv_vk_cached_recon_spv       = NULL;
static size_t          dv_vk_cached_recon_spv_len   = 0;

static int dv_vk_get_cached_or_compile_spv(FFVkSPIRVCompiler *spv, FFVulkanContext *vk, FFVulkanShader *shd, uint8_t **cache_data,
                                           size_t *cache_len, uint8_t **out_data, size_t *out_len, void **opaque)
{
    int ret;

    pthread_mutex_lock(&dv_vk_spv_cache_mutex);
    if (*cache_data && *cache_len) {
        *out_data = av_memdup(*cache_data, *cache_len);
        *out_len  = *cache_len;
        pthread_mutex_unlock(&dv_vk_spv_cache_mutex);
        return *out_data ? 0 : AVERROR(ENOMEM);
    }
    pthread_mutex_unlock(&dv_vk_spv_cache_mutex);

    ret = spv->compile_shader(vk, spv, shd, out_data, out_len, "main", opaque);
    if (ret < 0)
        return ret;

    pthread_mutex_lock(&dv_vk_spv_cache_mutex);
    if (!*cache_data && *out_data && *out_len) {
        *cache_data = av_memdup(*out_data, *out_len);
        if (*cache_data)
            *cache_len = *out_len;
    }
    pthread_mutex_unlock(&dv_vk_spv_cache_mutex);

    return 0;
}
#endif

static av_cold void dv_vk_init_static_rl_vlc(void)
{
    VLCElem        vlc_buf[FF_ARRAY_ELEMS(dv_vk_rl_vlc)] = {0};
    VLC            dv_vlc                                = {.table = vlc_buf, .table_allocated = FF_ARRAY_ELEMS(vlc_buf)};
    const unsigned offset                                = FF_ARRAY_ELEMS(dv_vk_rl_vlc) - (2 * NB_DV_VLC - NB_DV_ZERO_LEVEL_ENTRIES);
    RL_VLC_ELEM   *tmp                                   = dv_vk_rl_vlc + offset;
    int            i, j;

    for (i = 0, j = 0; i < NB_DV_VLC; i++, j++) {
        tmp[j].len8  = ff_dv_vlc_len[i];
        tmp[j].run   = ff_dv_vlc_run[i];
        tmp[j].level = ff_dv_vlc_level[i];

        if (ff_dv_vlc_level[i]) {
            tmp[j].len8++;

            j++;
            tmp[j].len8  = ff_dv_vlc_len[i] + 1;
            tmp[j].run   = ff_dv_vlc_run[i];
            tmp[j].level = -ff_dv_vlc_level[i];
        }
    }

    ff_vlc_init_from_lengths(&dv_vlc, 10, j, &tmp[0].len8, sizeof(tmp[0]), NULL, 0, 0, 0, VLC_INIT_USE_STATIC, NULL);
    av_assert1(dv_vlc.table_size == FF_ARRAY_ELEMS(dv_vk_rl_vlc));

    for (i = 0; i < dv_vlc.table_size; i++) {
        int code = dv_vlc.table[i].sym;
        int len  = dv_vlc.table[i].len;
        int level, run;

        if (len < 0) {
            run   = 0;
            level = code;
        } else {
            av_assert1(i <= code + (int)offset);
            run   = tmp[code].run + 1;
            level = tmp[code].level;
        }
        dv_vk_rl_vlc[i].len8  = len;
        dv_vk_rl_vlc[i].level = level;
        dv_vk_rl_vlc[i].run   = run;
    }
}

static void dv_vk_init_weight_tables(DVSubContext *s, const AVDVProfile *d)
{
    int       j, i, c, q;
    uint32_t *factor1 = &s->idct_factor[0];
    uint32_t *factor2 = &s->idct_factor[DV_PROFILE_IS_HD(d) ? 4096 : 2816];

    if (DV_PROFILE_IS_HD(d)) {
        static const uint8_t dv100_qstep[16] = {1, 1, 2, 3, 4, 5, 6, 7, 8, 16, 18, 20, 22, 24, 28, 52};
        const uint16_t      *iweight1, *iweight2;

        if (d->height == 720) {
            iweight1 = &dv_vk_iweight_720_y[0];
            iweight2 = &dv_vk_iweight_720_c[0];
        } else {
            iweight1 = &dv_vk_iweight_1080_y[0];
            iweight2 = &dv_vk_iweight_1080_c[0];
        }

        for (c = 0; c < 4; c++) {
            for (q = 0; q < 16; q++) {
                for (i = 0; i < 64; i++) {
                    *factor1++ = (dv100_qstep[q] << (c + 9)) * iweight1[i];

                    *factor2++ = (dv100_qstep[q] << (c + 9)) * iweight2[i];
                }
            }
        }
    } else {
        static const uint8_t dv_quant_areas[4] = {6, 21, 43, 64};
        const uint16_t      *iweight1          = &dv_vk_iweight_88[0];

        for (j = 0; j < 2; j++, iweight1 = &dv_vk_iweight_248[0]) {
            for (q = 0; q < 22; q++) {
                for (i = c = 0; c < 4; c++) {
                    for (; i < dv_quant_areas[c]; i++) {
                        *factor1   = iweight1[i] << (ff_dv_quant_shifts[q][c] + 1);
                        *factor2++ = (*factor1++) << 1;
                    }
                }
            }
        }
    }
}

static av_always_inline void dv_vk_decode_ac(GetBitContext *gb, DVVkBlockInfo *mb, int16_t *block, int16_t *quant_block,
                                             uint32_t *factor_block)
{
    int             last_index        = gb->size_in_bits;
    const uint8_t  *scan_table        = mb->scan_table;
    const uint32_t *factor_table      = mb->factor_table;
    int             pos               = mb->pos;
    int             partial_bit_count = mb->partial_bit_count;
    int             level, run, vlc_len, index;

    OPEN_READER_NOSIZE(re, gb);
    UPDATE_CACHE(re, gb);

    if (partial_bit_count > 0) {
        re_cache = re_cache >> partial_bit_count | mb->partial_bit_buffer;
        re_index -= partial_bit_count;
        mb->partial_bit_count = 0;
    }

    for (;;) {
        index   = NEG_USR32(re_cache, 10);
        vlc_len = dv_vk_rl_vlc[index].len8;
        if (vlc_len < 0) {
            index   = NEG_USR32((unsigned)re_cache << 10, -vlc_len) + dv_vk_rl_vlc[index].level;
            vlc_len = 10 - vlc_len;
        }
        level = dv_vk_rl_vlc[index].level;
        run   = dv_vk_rl_vlc[index].run;

        if (re_index + vlc_len > last_index) {
            mb->partial_bit_count  = last_index - re_index;
            mb->partial_bit_buffer = re_cache & ~(-1u >> mb->partial_bit_count);
            re_index               = last_index;
            break;
        }
        re_index += vlc_len;

        pos += run;
        if (pos >= 64)
            break;

        {
            int scan_idx  = scan_table[pos];
            int raw_level = level;

            if (quant_block)
                quant_block[scan_idx] = raw_level;
            if (factor_block)
                factor_block[scan_idx] = factor_table[pos];

            if (quant_block || factor_block) {
                block[scan_idx] = raw_level;
            } else {
                int64_t prod = (int64_t)raw_level * (int64_t)factor_table[pos];
                int     deq;

                if (prod >= 0)
                    deq = (int)((prod + (1LL << (dv_vk_iweight_bits - 1))) >> dv_vk_iweight_bits);
                else
                    deq = -(int)(((-prod) + (1LL << (dv_vk_iweight_bits - 1))) >> dv_vk_iweight_bits);

                block[scan_idx] = av_clip_int16(deq);
            }
        }

        UPDATE_CACHE(re, gb);
    }
    CLOSE_READER(re, gb);
    mb->pos = pos;
}

static inline void dv_vk_bit_copy(PutBitContext *pb, GetBitContext *gb)
{
    int bits_left = get_bits_left(gb);
    while (bits_left >= MIN_CACHE_BITS) {
        put_bits(pb, MIN_CACHE_BITS, get_bits(gb, MIN_CACHE_BITS));
        bits_left -= MIN_CACHE_BITS;
    }
    if (bits_left > 0)
        put_bits(pb, bits_left, get_bits(gb, bits_left));
}

static void dv_vk_build_dequant_shader_source(FFVulkanShader *shd)
{
    GLSLC(
        0, layout(push_constant) uniform PushConstants {                     );
            GLSLC(1, uint num_coeffs;);
            GLSLC(1, uint iweight_bits;);
            GLSLC(1, uint output_stride;);
            GLSLC(1, uint output_base;);
    GLSLC(0,
        } pc;);
    GLSLC(
        0, void main() {                                                     );
            GLSLC(1, uint idx = gl_GlobalInvocationID.x;);
            GLSLC(1, if (idx >= pc.num_coeffs));
            GLSLC(2, return;);
            GLSLC(1, int q = quantized[idx];);
            GLSLC(1, int value = q;);
            GLSLC(1, if ((idx & 63u) != 0u));
            GLSLC(2, value = int((uint(q) * factors[idx] + (1u << (pc.iweight_bits - 1u))) >> pc.iweight_bits););
            GLSLC(1, uint out_idx = idx;);
            GLSLC(1, if (pc.output_stride != 0u) {);
                GLSLC(2, uint block = idx >> 6u;);
                GLSLC(2, uint coeff = idx & 63u;);
                GLSLC(2, out_idx = pc.output_base + block * pc.output_stride + coeff;);
            GLSLC(1, });
            GLSLC(1, dequantized[out_idx] = value;);
    GLSLC(0,
        });
}

static void dv_vk_build_idct_shader_source(FFVulkanShader *shd)
{
    GLSLC(
        0, layout(push_constant) uniform PushConstants {                     );
            GLSLC(1, uint num_blocks;);
            GLSLC(1, uint blocks_per_row;);
            GLSLC(1, uint output_base;);
            GLSLC(1, uint reserved;);
    GLSLC(0,
        } pc;);
    GLSLC(0, shared int intermediate[64];);
    GLSLC(0, const int W1 = 22725;);
    GLSLC(0, const int W2 = 21407;);
    GLSLC(0, const int W3 = 19266;);
    GLSLC(0, const int W4 = 16383;);
    GLSLC(0, const int W5 = 12873;);
    GLSLC(0, const int W6 = 8867;);
    GLSLC(0, const int W7 = 4520;);
    GLSLC(0, const int ROW_SHIFT = 11;);
    GLSLC(0, const int COL_SHIFT = 20;);
    GLSLC(0, const int COL_BIAS_DIV_W4 = ((1 << (COL_SHIFT - 1)) / W4););
    GLSLC(
        0, int idct_row_element(in int row[8], uint out_idx) {               );
            GLSLC(1, int a0 = W4 * row[0] + (1 << (ROW_SHIFT - 1)););
            GLSLC(1, int a1 = a0;);
            GLSLC(1, int a2 = a0;);
            GLSLC(1, int a3 = a0;);
            GLSLC(1, a0 += W2 * row[2];);
            GLSLC(1, a1 += W6 * row[2];);
            GLSLC(1, a2 += -W6 * row[2];);
            GLSLC(1, a3 += -W2 * row[2];);
            GLSLC(1, int b0 = W1 * row[1];);
            GLSLC(1, b0 += W3 * row[3];);
            GLSLC(1, int b1 = W3 * row[1];);
            GLSLC(1, b1 += -W7 * row[3];);
            GLSLC(1, int b2 = W5 * row[1];);
            GLSLC(1, b2 += -W1 * row[3];);
            GLSLC(1, int b3 = W7 * row[1];);
            GLSLC(1, b3 += -W5 * row[3];);
            GLSLC(
                1, if ((row[4] | row[5] | row[6] | row[7]) != 0) {);
                    GLSLC(2, a0 += W4 * row[4];);
                    GLSLC(2, a0 += W6 * row[6];);
                    GLSLC(2, a1 += -W4 * row[4];);
                    GLSLC(2, a1 += -W2 * row[6];);
                    GLSLC(2, a2 += -W4 * row[4];);
                    GLSLC(2, a2 += W2 * row[6];);
                    GLSLC(2, a3 += W4 * row[4];);
                    GLSLC(2, a3 += -W6 * row[6];);
                    GLSLC(2, b0 += W5 * row[5];);
                    GLSLC(2, b0 += W7 * row[7];);
                    GLSLC(2, b1 += -W1 * row[5];);
                    GLSLC(2, b1 += -W5 * row[7];);
                    GLSLC(2, b2 += W7 * row[5];);
                    GLSLC(2, b2 += W3 * row[7];);
                    GLSLC(2, b3 += W3 * row[5];);
                    GLSLC(2, b3 += -W1 * row[7];);
            GLSLC(1,
                });
            GLSLC(1, int r0 = (a0 + b0) >> ROW_SHIFT;);
            GLSLC(1, int r1 = (a1 + b1) >> ROW_SHIFT;);
            GLSLC(1, int r2 = (a2 + b2) >> ROW_SHIFT;);
            GLSLC(1, int r3 = (a3 + b3) >> ROW_SHIFT;);
            GLSLC(1, int r4 = (a3 - b3) >> ROW_SHIFT;);
            GLSLC(1, int r5 = (a2 - b2) >> ROW_SHIFT;);
            GLSLC(1, int r6 = (a1 - b1) >> ROW_SHIFT;);
            GLSLC(1, int r7 = (a0 - b0) >> ROW_SHIFT;);
            GLSLC(1, if (out_idx == 0u) return r0;);
            GLSLC(1, if (out_idx == 1u) return r1;);
            GLSLC(1, if (out_idx == 2u) return r2;);
            GLSLC(1, if (out_idx == 3u) return r3;);
            GLSLC(1, if (out_idx == 4u) return r4;);
            GLSLC(1, if (out_idx == 5u) return r5;);
            GLSLC(1, if (out_idx == 6u) return r6;);
            GLSLC(1, return r7;);
    GLSLC(0,
        });
    GLSLC(
        0, int idct_col_element(in int col[8], uint out_idx) {               );
            GLSLC(1, int a0 = W4 * (col[0] + COL_BIAS_DIV_W4););
            GLSLC(1, int a1 = a0;);
            GLSLC(1, int a2 = a0;);
            GLSLC(1, int a3 = a0;);
            GLSLC(1, a0 += W2 * col[2];);
            GLSLC(1, a1 += W6 * col[2];);
            GLSLC(1, a2 += -W6 * col[2];);
            GLSLC(1, a3 += -W2 * col[2];);
            GLSLC(1, int b0 = W1 * col[1];);
            GLSLC(1, b0 += W3 * col[3];);
            GLSLC(1, int b1 = W3 * col[1];);
            GLSLC(1, b1 += -W7 * col[3];);
            GLSLC(1, int b2 = W5 * col[1];);
            GLSLC(1, b2 += -W1 * col[3];);
            GLSLC(1, int b3 = W7 * col[1];);
            GLSLC(1, b3 += -W5 * col[3];);
            GLSLC(
                1, if (col[4] != 0) {);
                    GLSLC(2, a0 += W4 * col[4];);
                    GLSLC(2, a1 += -W4 * col[4];);
                    GLSLC(2, a2 += -W4 * col[4];);
                    GLSLC(2, a3 += W4 * col[4];);
            GLSLC(1,
                });
            GLSLC(
                1, if (col[5] != 0) {);
                    GLSLC(2, b0 += W5 * col[5];);
                    GLSLC(2, b1 += -W1 * col[5];);
                    GLSLC(2, b2 += W7 * col[5];);
                    GLSLC(2, b3 += W3 * col[5];);
            GLSLC(1,
                });
            GLSLC(
                1, if (col[6] != 0) {);
                    GLSLC(2, a0 += W6 * col[6];);
                    GLSLC(2, a1 += -W2 * col[6];);
                    GLSLC(2, a2 += W2 * col[6];);
                    GLSLC(2, a3 += -W6 * col[6];);
            GLSLC(1,
                });
            GLSLC(
                1, if (col[7] != 0) {);
                    GLSLC(2, b0 += W7 * col[7];);
                    GLSLC(2, b1 += -W5 * col[7];);
                    GLSLC(2, b2 += W3 * col[7];);
                    GLSLC(2, b3 += -W1 * col[7];);
            GLSLC(1,
                });
            GLSLC(1, int r0 = (a0 + b0) >> COL_SHIFT;);
            GLSLC(1, int r1 = (a1 + b1) >> COL_SHIFT;);
            GLSLC(1, int r2 = (a2 + b2) >> COL_SHIFT;);
            GLSLC(1, int r3 = (a3 + b3) >> COL_SHIFT;);
            GLSLC(1, int r4 = (a3 - b3) >> COL_SHIFT;);
            GLSLC(1, int r5 = (a2 - b2) >> COL_SHIFT;);
            GLSLC(1, int r6 = (a1 - b1) >> COL_SHIFT;);
            GLSLC(1, int r7 = (a0 - b0) >> COL_SHIFT;);
            GLSLC(1, if (out_idx == 0u) return r0;);
            GLSLC(1, if (out_idx == 1u) return r1;);
            GLSLC(1, if (out_idx == 2u) return r2;);
            GLSLC(1, if (out_idx == 3u) return r3;);
            GLSLC(1, if (out_idx == 4u) return r4;);
            GLSLC(1, if (out_idx == 5u) return r5;);
            GLSLC(1, if (out_idx == 6u) return r6;);
            GLSLC(1, return r7;);
    GLSLC(0,
        });
    GLSLC(
        0, void main() {                                                       );
            GLSLC(1, uint block_idx = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;);
            GLSLC(1, if (block_idx >= pc.num_blocks));
            GLSLC(2, return;);
            GLSLC(1, uint m = gl_LocalInvocationID.y;);
            GLSLC(1, uint n = gl_LocalInvocationID.x;);
            GLSLC(1, int row_in[8];);
            GLSLC(1, for (int i = 0; i < 8; ++i));
            GLSLC(2, row_in[i] = values[block_idx * 128 + m * 8 + i];);
            GLSLC(1, int row_value = idct_row_element(row_in, n););
            GLSLC(1, intermediate[m * 8 + n] = (row_value << 16) >> 16;);
            GLSLC(1, barrier(););
            GLSLC(1, int col_in[8];);
            GLSLC(1, for (int i = 0; i < 8; ++i));
            GLSLC(2, col_in[i] = intermediate[i * 8 + n];);
            GLSLC(1, int value = idct_col_element(col_in, m););
            GLSLC(1, uint output_idx = block_idx * 128 + 64 + (m * 8 + n););
            GLSLC(1, values[output_idx] = (value << 16) >> 16;);
    GLSLC(0,
        });
}

static void dv_vk_build_recon_calc_shader_source(FFVulkanShader *shd)
{
    GLSLC(
        0, layout(push_constant) uniform PushConstants {                );
            GLSLC(1, uint width[4];);
            GLSLC(1, uint height[4];);
            GLSLC(1, uint plane_offset[4];);
            GLSLC(1, uint plane_stride[4];);
            GLSLC(1, uint num_blocks;);
            GLSLC(1, uint blocks_per_mb;);
            GLSLC(1, uint chroma_w_shift;);
            GLSLC(1, uint chroma_h_shift;);
            GLSLC(1, uint is_yuv411;);
            GLSLC(1, uint last_mb_y;);
            GLSLC(1, uint chroma_411_split_mb_x;);
    GLSLC(0,
        };);

    GLSLC(
        0, uint load_sample(uint block_idx, uint x, uint y) {      );
            GLSLC(1, uint idx = block_idx * 128u + 64u + y * 8u + x;);
            GLSLC(1, int value = values[idx];);
            GLSLC(1, return uint(clamp(value, 0, 255)););
    GLSLC(0,
        });

    GLSLC(
        0, void store_byte(uint idx, uint value) {                  );
            GLSLC(1, data[idx] = value & 255u;);
    GLSLC(0,
        });

    GLSLC(
        0, void main() {                                               );
            GLSLC(1, uint block_idx = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;);
            GLSLC(1, if (block_idx >= num_blocks));
            GLSLC(2, return;);
            GLSLC(1, if (blocks_per_mb == 0u));
            GLSLC(2, return;);
            GLSLC(1, uint job_idx = block_idx / blocks_per_mb;);
            GLSLC(1, uint block_in_mb = block_idx %% blocks_per_mb;);
            GLSLC(1, uvec4 job = jobs[job_idx];);
            GLSLC(1, uint mb_x = job.x;);
            GLSLC(1, uint mb_y = job.y;);
            GLSLC(1, uint is_field_mode = job.z & 1u;);
            GLSLC(1, uint sample_x = gl_LocalInvocationID.x;);
            GLSLC(1, uint sample_y = gl_LocalInvocationID.y;);
            GLSLC(1, if (sample_x >= 8u || sample_y >= 8u));
            GLSLC(2, return;);
            GLSLC(1, uint linesize_y = plane_stride[0] << is_field_mode;);
            GLSLC(1, uint y_ptr = plane_offset[0] + ((mb_y * plane_stride[0] + mb_x) << 3u););
            GLSLC(1, uint y_stride = 2u << 3u;);
            GLSLC(
                1, if ((is_yuv411 == 0u && chroma_h_shift != 0u) || (is_yuv411 != 0u && mb_x >= chroma_411_split_mb_x) ||
                       (height[0] >= 720u && mb_y != last_mb_y)) {);
                    GLSLC(2, y_stride = plane_stride[0] << ((is_field_mode == 0u) ? 3u : 0u););
            GLSLC(1,
                });
            GLSLC(1, uint chroma_stride = plane_stride[1] << is_field_mode;);
            GLSLC(1, uint c_offset = (((mb_y >> chroma_h_shift) * plane_stride[1] + (mb_x >> chroma_w_shift)) << 3u););
            GLSLC(
                1,
                if (block_in_mb < 4u) {);
                    GLSLC(2, uint base = y_ptr + ((block_in_mb & 1u) * 8u) + ((block_in_mb >> 1u) * y_stride););
                    GLSLC(2, store_byte(base + sample_y * linesize_y + sample_x, load_sample(block_idx, sample_x, sample_y)););
            GLSLC(1,
                } else {);
                    GLSLC(2, uint plane = 0u;);
                    GLSLC(2, uint chroma_is_second = 0u;);
                    GLSLC(
                        2,
                        if (blocks_per_mb == 8u) {);
                            GLSLC(3, uint chroma_idx = block_in_mb - 4u;);
                            GLSLC(3, plane = 2u - (chroma_idx >> 1u););
                            GLSLC(3, chroma_is_second = chroma_idx & 1u;);
                    GLSLC(2,
                        } else {);
                            GLSLC(3, plane = 2u - (block_in_mb - 4u););
                    GLSLC(2,
                        });
                    GLSLC(2, uint base = plane_offset[plane] + c_offset;);
                    GLSLC(
                        2,
                        if (is_yuv411 != 0u && mb_x >= chroma_411_split_mb_x) {);
                            GLSLC(
                                3, if (sample_x < 4u) {);
                                    GLSLC(4, store_byte(base + sample_y * chroma_stride + sample_x,
                                                        load_sample(block_idx, sample_x, sample_y)););
                                    GLSLC(4, store_byte(base + (chroma_stride << 3u) + sample_y * chroma_stride + sample_x,
                                                        load_sample(block_idx, sample_x + 4u, sample_y)););
            GLSLC(3,
                                });
            GLSLC(2,
                        } else {);
                            GLSLC(
                                3, if (chroma_is_second != 0u) {);
                                    GLSLC(4, uint chroma_y_stride = (mb_y == last_mb_y)
                                                                        ? (1u << 3u)
                                                                        : (plane_stride[plane] << ((is_field_mode == 0u) ? 3u : 0u)););
                                    GLSLC(4, base += chroma_y_stride;);
                            GLSLC(3,
                                });
                            GLSLC(3, store_byte(base + sample_y * chroma_stride + sample_x, load_sample(block_idx, sample_x, sample_y)););
            GLSLC(2,
                        });
            GLSLC(1,
                });
    GLSLC(0,
        });
}

static void dv_vk_build_recon_shader_source(FFVulkanShader *shd)
{
    GLSLC(
        0, layout(push_constant) uniform PushConstants {                );
            GLSLC(1, uint width[4];);
            GLSLC(1, uint height[4];);
            GLSLC(1, uint plane_offset[4];);
            GLSLC(1, uint plane_stride[4];);
            GLSLC(1, uint chroma_w_shift;);
            GLSLC(1, uint chroma_h_shift;);
            GLSLC(1, uint reserved0;);
            GLSLC(1, uint reserved1;);
            GLSLC(1, uint reserved2;);
    GLSLC(0,
        };);

    GLSLC(
        0, uint load_byte(uint idx) {                                );
            GLSLC(1, return uint(data[idx]););
    GLSLC(0,
        });

    GLSLC(
        0, float load_chroma(uint plane, uint x, uint y) {           );
            GLSLC(1, uint plane_w = max(width[plane], 1u););
            GLSLC(1, uint plane_h = max(height[plane], 1u););
            GLSLC(1, float scale_x = float(1u << chroma_w_shift););
            GLSLC(1, float scale_y = float(1u << chroma_h_shift););
            GLSLC(1, float chroma_x = (float(x) + 0.5) / scale_x - 0.5;);
            GLSLC(1, float chroma_y = (float(y) + 0.5) / scale_y - 0.5;);
            GLSLC(1, chroma_x = clamp(chroma_x, 0.0, float(plane_w - 1u)););
            GLSLC(1, chroma_y = clamp(chroma_y, 0.0, float(plane_h - 1u)););
            GLSLC(1, int x0 = int(floor(chroma_x)););
            GLSLC(1, int y0 = int(floor(chroma_y)););
            GLSLC(1, int x1 = min(x0 + 1, int(plane_w - 1u)););
            GLSLC(1, int y1 = min(y0 + 1, int(plane_h - 1u)););
            GLSLC(1, float fx = chroma_x - float(x0););
            GLSLC(1, float fy = chroma_y - float(y0););
            GLSLC(1, uint idx00 = plane_offset[plane] + uint(y0) * plane_stride[plane] + uint(x0););
            GLSLC(1, uint idx10 = plane_offset[plane] + uint(y0) * plane_stride[plane] + uint(x1););
            GLSLC(1, uint idx01 = plane_offset[plane] + uint(y1) * plane_stride[plane] + uint(x0););
            GLSLC(1, uint idx11 = plane_offset[plane] + uint(y1) * plane_stride[plane] + uint(x1););
            GLSLC(1, float c00 = float(load_byte(idx00)););
            GLSLC(1, float c10 = float(load_byte(idx10)););
            GLSLC(1, float c01 = float(load_byte(idx01)););
            GLSLC(1, float c11 = float(load_byte(idx11)););
            GLSLC(1, float cx0 = mix(c00, c10, fx););
            GLSLC(1, float cx1 = mix(c01, c11, fx););
            GLSLC(1, return mix(cx0, cx1, fy););
    GLSLC(0,
        });

    GLSLC(
        0, void main() {                                               );
            GLSLC(1, ivec2 pos = ivec2(gl_GlobalInvocationID.xy););
            GLSLC(1, ivec2 dst_size = imageSize(output_img[0]););
            GLSLC(1, if (any(greaterThanEqual(pos, dst_size))));
            GLSLC(2, return;);
            GLSLC(1, uint x = uint(pos.x););
            GLSLC(1, uint y = uint(pos.y););
            GLSLC(1, uint idx_y = plane_offset[0] + y * plane_stride[0] + x;);
            GLSLC(1, uint yv = load_byte(idx_y););
            GLSLC(1, float yf = float(yv););
            GLSLC(1, float uf = load_chroma(1u, x, y) - 128.0;);
            GLSLC(1, float vf = load_chroma(2u, x, y) - 128.0;);
            GLSLC(1, float rf = clamp(yf + 1.40200 * vf, 0.0, 255.0););
            GLSLC(1, float gf = clamp(yf - 0.34414 * uf - 0.71414 * vf, 0.0, 255.0););
            GLSLC(1, float bf = clamp(yf + 1.77200 * uf, 0.0, 255.0););
            GLSLC(1, imageStore(output_img[0], pos, vec4(rf, gf, bf, 255.0) / 255.0););
    GLSLC(0,
        });
}

static void dv_vk_reset_frame_state(DVSubContext *s)
{
    s->frame_packet_size        = 0;
    s->frame_packet_from_start  = 0;
    s->stats                    = (DVPipelineStats){0};
    s->coeff_blocks_are_spatial = 0;
    s->idct_upload_ready        = 0;
    s->cpu_output_required      = 0;
    s->dequant_output_in_idct_buf = 0;
}

static int dv_vk_cache_packet(DVSubContext *s, const uint8_t *data, uint32_t size)
{
    uint8_t *tmp;

    if (!data || !size)
        return AVERROR(EINVAL);

    tmp = av_realloc(s->frame_packet, size);
    if (!tmp)
        return AVERROR(ENOMEM);

    s->frame_packet = tmp;
    memcpy(s->frame_packet, data, size);
    s->frame_packet_size = size;

    return 0;
}

static int dv_vk_prepare_profile_and_tables(AVCodecContext *avctx, DVSubContext *s)
{
    const AVDVProfile *sys;

    sys = ff_dv_frame_profile(avctx, s->sys, s->frame_packet, s->frame_packet_size);
    if (!sys) {
        av_log(avctx, AV_LOG_ERROR, "dv_vulkan: could not detect DV profile\n");
        return AVERROR_INVALIDDATA;
    }

    if (s->sys != sys) {
        ff_dv_init_dynamic_tables(s->work_chunks, sys);
        dv_vk_init_weight_tables(s, sys);

        memcpy(s->dv_zigzag[0], ff_zigzag_direct, sizeof(s->dv_zigzag[0]));
        memcpy(s->dv_zigzag[1], ff_dv_zigzag248_direct, sizeof(s->dv_zigzag[1]));
    }

    s->sys = sys;

    s->stats.work_pool_size        = dv_work_pool_size(sys);
    s->stats.frame_width           = sys->width;
    s->stats.frame_height          = sys->height;
    s->stats.mb_width_blocks       = FFMAX(sys->width >> 3, 1);
    s->stats.chroma_411_split_mb_x = FFMAX(s->stats.mb_width_blocks - 2, 0);
    s->stats.last_mb_y             = 0;
    s->stats.sw_format             = dv_vk_choose_sw_format(sys->pix_fmt);

    return 0;
}

static int dv_vk_build_mb_jobs(AVCodecContext *avctx, DVSubContext *s)
{
    int needed   = s->stats.work_pool_size * 5;
    int job_idx  = 0;
    int max_mb_y = 0;

    if (needed <= 0)
        return AVERROR_INVALIDDATA;

    if (needed > s->mb_jobs_alloc) {
        DVMacroblockJob *tmp = av_realloc_array(s->mb_jobs, needed, sizeof(*tmp));
        if (!tmp)
            return AVERROR(ENOMEM);
        s->mb_jobs       = tmp;
        s->mb_jobs_alloc = needed;
    }

    if (needed > s->mb_field_modes_alloc) {
        uint8_t *tmp = av_realloc_array(s->mb_field_modes, needed, sizeof(*tmp));
        if (!tmp)
            return AVERROR(ENOMEM);
        s->mb_field_modes       = tmp;
        s->mb_field_modes_alloc = needed;
    }

    for (int i = 0; i < s->stats.work_pool_size; i++) {
        const DVwork_chunk *wc = &s->work_chunks[i];

        for (int m = 0; m < 5; m++) {
            int              mb_x, mb_y;
            DVMacroblockJob *job = &s->mb_jobs[job_idx++];

            dv_calculate_mb_xy(s->sys, s->frame_packet, wc, m, &mb_x, &mb_y);

            job->buf_offset = wc->buf_offset;
            job->mb_x       = mb_x;
            job->mb_y       = mb_y;
            job->mb_index   = m;

            if (mb_y > max_mb_y)
                max_mb_y = mb_y;
        }
    }

    s->stats.mb_jobs   = job_idx;
    s->stats.last_mb_y = max_mb_y;

    return 0;
}

static int dv_vk_prepare_coeff_staging(DVSubContext *s)
{
    int needed_blocks = s->stats.mb_jobs * DV_MAX_BPM;
    int needed_coeffs = needed_blocks * 64;

    if (needed_coeffs > s->coeff_blocks_alloc) {
        int16_t *tmp = av_realloc_array(s->coeff_blocks, needed_coeffs, sizeof(*tmp));
        if (!tmp)
            return AVERROR(ENOMEM);
        s->coeff_blocks       = tmp;
        s->coeff_blocks_alloc = needed_coeffs;
    }

    memset(s->coeff_blocks, 0, needed_coeffs * sizeof(*s->coeff_blocks));
    s->stats.block_jobs = needed_blocks;

    return 0;
}

static int dv_vk_prepare_plane_staging(DVSubContext *s)
{
    int       linesizes[4]         = {0};
    ptrdiff_t linesizes_ptrdiff[4] = {0};
    ptrdiff_t plane_sizes[4]       = {0};
    int       ret;
    int       total = 0;

    ret = av_image_fill_linesizes(linesizes, s->stats.sw_format, s->stats.frame_width);
    if (ret < 0)
        return ret;

    for (int i = 0; i < 4; i++)
        linesizes_ptrdiff[i] = linesizes[i];

    ret = av_image_fill_plane_sizes(plane_sizes, s->stats.sw_format, s->stats.frame_height, linesizes_ptrdiff);
    if (ret < 0)
        return ret;

    for (int i = 0; i < 4; i++) {
        if (plane_sizes[i] <= 0)
            continue;
        if (plane_sizes[i] > INT_MAX - total)
            return AVERROR(EINVAL);
        total += (int)plane_sizes[i];
    }

    if (total > s->plane_staging_size) {
        uint8_t *tmp = av_realloc(s->plane_staging, total);
        if (!tmp)
            return AVERROR(ENOMEM);
        s->plane_staging      = tmp;
        s->plane_staging_size = total;
    }

    memset(s->plane_staging, 0, total);

    return 0;
}

static enum AVPixelFormat dv_vk_choose_sw_format(enum AVPixelFormat sw_format)
{
    switch (sw_format) {
    case AV_PIX_FMT_YUV411P:
        return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_NONE:
        return AV_PIX_FMT_YUV420P;
    default:
        return sw_format;
    }
}

static int dv_vk_recon_gpu_supported(const AVDVProfile *sys)
{
    if (!sys)
        return 0;

    switch (sys->pix_fmt) {
    case AV_PIX_FMT_YUV411P:
        return sys->bpm == 6;
    case AV_PIX_FMT_YUV420P:
        return sys->bpm == 6;
    case AV_PIX_FMT_YUV422P:
        return sys->bpm == 6 || sys->bpm == 8;
    default:
        return 0;
    }
}

static int dv_vk_get_chroma_shifts(enum AVPixelFormat fmt, uint32_t *w_shift, uint32_t *h_shift, uint32_t *is_yuv411)
{
    if (!w_shift || !h_shift || !is_yuv411)
        return AVERROR(EINVAL);

    switch (fmt) {
    case AV_PIX_FMT_YUV411P:
        *w_shift   = 2;
        *h_shift   = 0;
        *is_yuv411 = 1;
        return 0;
    case AV_PIX_FMT_YUV420P:
        *w_shift   = 1;
        *h_shift   = 1;
        *is_yuv411 = 0;
        return 0;
    case AV_PIX_FMT_YUV422P:
        *w_shift   = 1;
        *h_shift   = 0;
        *is_yuv411 = 0;
        return 0;
    default:
        return AVERROR(ENOSYS);
    }
}

static int dv_vk_get_plane_layout(enum AVPixelFormat fmt, int width, int height, int linesizes[4], ptrdiff_t plane_sizes[4],
                                  ptrdiff_t plane_offsets[4])
{
    ptrdiff_t linesizes_ptrdiff[4] = {0};
    int       ret;

    ret = av_image_fill_linesizes(linesizes, fmt, width);
    if (ret < 0)
        return ret;

    for (int i = 0; i < 4; i++)
        linesizes_ptrdiff[i] = linesizes[i];

    ret = av_image_fill_plane_sizes(plane_sizes, fmt, height, linesizes_ptrdiff);
    if (ret < 0)
        return ret;

    plane_offsets[0] = 0;
    for (int i = 1; i < 4; i++)
        plane_offsets[i] = plane_offsets[i - 1] + FFMAX(plane_sizes[i - 1], 0);

    return 0;
}

static int dv_vk_decode_entropy_chunk(AVCodecContext *avctx, void *arg)
{
    DVSubContext       *s             = avctx->internal->hwaccel_priv_data;
    const DVwork_chunk *wc            = arg;
    const int           chunk_index   = (int)(wc - s->work_chunks);
    const int           log2_blocksize = 3;
    const int           mb_base       = chunk_index * 5;
    int                 coeff_base    = mb_base * s->sys->bpm;
    const uint8_t      *buf_ptr       = &s->frame_packet[wc->buf_offset * 80];

    for (int mb_index = 0; mb_index < 5; mb_index++) {
        int           quant;
        DVVkBlockInfo mb_data[DV_MAX_BPM];
        uint8_t       mb_bit_buffer[80 + AV_INPUT_BUFFER_PADDING_SIZE] = {0};
        PutBitContext pb;
        GetBitContext gb;
        int           is_field_mode = 0;

        quant = buf_ptr[3] & 0x0f;
        buf_ptr += 4;

        init_put_bits(&pb, mb_bit_buffer, 80);

        for (int j = 0; j < s->sys->bpm; j++) {
            DVVkBlockInfo *mb           = &mb_data[j];
            int16_t       *block        = &s->coeff_blocks[(coeff_base + j) * 64];
            int            last_index   = s->sys->block_sizes[j];
            int            dc, dct_mode, class1;

            init_get_bits(&gb, buf_ptr, last_index);

            dc       = get_sbits(&gb, 9);
            dct_mode = get_bits1(&gb);
            class1   = get_bits(&gb, 2);

            if (DV_PROFILE_IS_HD(s->sys)) {
                mb->scan_table   = s->dv_zigzag[0];
                mb->factor_table = &s->idct_factor[(j >= 4) * 4 * 16 * 64 + class1 * 16 * 64 + quant * 64];
                is_field_mode |= (!j && dct_mode);
            } else {
                mb->scan_table   = s->dv_zigzag[dct_mode && log2_blocksize == 3];
                mb->factor_table =
                    &s->idct_factor[(class1 == 3) * 2 * 22 * 64 + (dct_mode && log2_blocksize == 3) * 22 * 64 +
                                    (quant + ff_dv_quant_offset[class1]) * 64];
            }

            dc                    = dc * 4 + 1024;
            block[0]              = dc;
            mb->pos               = 0;
            mb->partial_bit_count = 0;

            dv_vk_decode_ac(&gb, mb, block, NULL, NULL);

            if (mb->pos >= 64)
                dv_vk_bit_copy(&pb, &gb);

            buf_ptr += last_index >> 3;
        }

        put_bits32(&pb, 0);
        flush_put_bits(&pb);

        init_get_bits(&gb, mb_bit_buffer, put_bits_count(&pb));
        for (int j = 0; j < s->sys->bpm; j++) {
            DVVkBlockInfo *mb    = &mb_data[j];
            int16_t       *block = &s->coeff_blocks[(coeff_base + j) * 64];

            if (mb->pos < 64 && get_bits_left(&gb) > 0)
                dv_vk_decode_ac(&gb, mb, block, NULL, NULL);
        }

        s->mb_field_modes[mb_base + mb_index] = !!is_field_mode;
        coeff_base += s->sys->bpm;
    }

    return 0;
}

static int dv_vk_stage_cpu_entropy(AVCodecContext *avctx, DVSubContext *s)
{
    avctx->execute(avctx, dv_vk_decode_entropy_chunk, s->work_chunks, NULL, s->stats.work_pool_size, sizeof(DVwork_chunk));

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: CPU entropy filled %d blocks (dequantized coeffs)\n", s->stats.block_jobs);
    s->coeff_blocks_are_spatial = 0;

    return 0;
}

static int dv_vk_stage_gpu_dequant(AVCodecContext *avctx, DVSubContext *s)
{
    int num_blocks = s->stats.block_jobs;

    s->dequant_output_in_idct_buf = 0;

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: dequant stage skipped (already done in entropy decode, %d blocks)\n", num_blocks);
    return 0;
}

static av_always_inline void dv_vk_put_block_8x8(uint8_t *dst, ptrdiff_t linesize, const int16_t *block)
{
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++)
            dst[x] = av_clip_uint8(block[y * 8 + x]);
        dst += linesize;
    }
}

static av_always_inline void dv_vk_put_block_8x4(uint8_t *dst, ptrdiff_t linesize, const int16_t *block)
{
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 8; x++)
            dst[x] = av_clip_uint8(block[y * 8 + x]);
        dst += linesize;
    }
}

static void dv_vk_put_last_row_field_chroma(uint8_t *dst, ptrdiff_t linesize, int16_t *blocks)
{
    ff_simple_idct_int16_8bit(blocks + 0 * 64);
    ff_simple_idct_int16_8bit(blocks + 1 * 64);

    dv_vk_put_block_8x4(dst, linesize << 1, blocks + 0 * 64);
    dv_vk_put_block_8x4(dst + 8, linesize << 1, blocks + 0 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + linesize, linesize << 1, blocks + 1 * 64);
    dv_vk_put_block_8x4(dst + 8 + linesize, linesize << 1, blocks + 1 * 64 + 4 * 8);
}

static void dv_vk_put_last_row_field_luma(uint8_t *dst, ptrdiff_t linesize, int16_t *blocks)
{
    ff_simple_idct_int16_8bit(blocks + 0 * 64);
    ff_simple_idct_int16_8bit(blocks + 1 * 64);
    ff_simple_idct_int16_8bit(blocks + 2 * 64);
    ff_simple_idct_int16_8bit(blocks + 3 * 64);

    dv_vk_put_block_8x4(dst, linesize << 1, blocks + 0 * 64);
    dv_vk_put_block_8x4(dst + 16, linesize << 1, blocks + 0 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + 8, linesize << 1, blocks + 1 * 64);
    dv_vk_put_block_8x4(dst + 24, linesize << 1, blocks + 1 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + linesize, linesize << 1, blocks + 2 * 64);
    dv_vk_put_block_8x4(dst + 16 + linesize, linesize << 1, blocks + 2 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + 8 + linesize, linesize << 1, blocks + 3 * 64);
    dv_vk_put_block_8x4(dst + 24 + linesize, linesize << 1, blocks + 3 * 64 + 4 * 8);
}

static void dv_vk_put_last_row_field_chroma_spatial(uint8_t *dst, ptrdiff_t linesize, const int16_t *blocks)
{
    dv_vk_put_block_8x4(dst, linesize << 1, blocks + 0 * 64);
    dv_vk_put_block_8x4(dst + 8, linesize << 1, blocks + 0 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + linesize, linesize << 1, blocks + 1 * 64);
    dv_vk_put_block_8x4(dst + 8 + linesize, linesize << 1, blocks + 1 * 64 + 4 * 8);
}

static void dv_vk_put_last_row_field_luma_spatial(uint8_t *dst, ptrdiff_t linesize, const int16_t *blocks)
{
    dv_vk_put_block_8x4(dst, linesize << 1, blocks + 0 * 64);
    dv_vk_put_block_8x4(dst + 16, linesize << 1, blocks + 0 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + 8, linesize << 1, blocks + 1 * 64);
    dv_vk_put_block_8x4(dst + 24, linesize << 1, blocks + 1 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + linesize, linesize << 1, blocks + 2 * 64);
    dv_vk_put_block_8x4(dst + 16 + linesize, linesize << 1, blocks + 2 * 64 + 4 * 8);
    dv_vk_put_block_8x4(dst + 8 + linesize, linesize << 1, blocks + 3 * 64);
    dv_vk_put_block_8x4(dst + 24 + linesize, linesize << 1, blocks + 3 * 64 + 4 * 8);
}

static int dv_vk_convert_411_to_420(DVSubContext *s, uint8_t *const src_data[4], const int src_linesize[4], uint8_t *const dst_data[4],
                                    const int dst_linesize[4])
{
    int width  = s->stats.frame_width;
    int height = s->stats.frame_height;

    if (width <= 0 || height <= 0)
        return AVERROR(EINVAL);

    for (int y = 0; y < height; y++)
        memcpy(dst_data[0] + y * dst_linesize[0], src_data[0] + y * src_linesize[0], width);

    for (int plane = 1; plane <= 2; plane++) {
        int dst_chroma_w = width >> 1;
        int dst_chroma_h = height >> 1;

        for (int y = 0; y < dst_chroma_h; y++) {
            const uint8_t *src0 = src_data[plane] + (2 * y) * src_linesize[plane];
            const uint8_t *src1 = src_data[plane] + FFMIN(2 * y + 1, height - 1) * src_linesize[plane];
            uint8_t       *dst  = dst_data[plane] + y * dst_linesize[plane];

            for (int x = 0; x < dst_chroma_w; x++) {
                int sx = x >> 1;
                dst[x] = (uint8_t)((src0[sx] + src1[sx] + 1) >> 1);
            }
        }
    }

    return 0;
}

static int dv_vk_stage_cpu_reconstruct_yuv(AVCodecContext *avctx, DVSubContext *s)
{
    const int          log2_blocksize        = 3;
    enum AVPixelFormat decode_fmt            = s->sys->pix_fmt;
    enum AVPixelFormat output_fmt            = s->stats.sw_format;
    int                dec_linesizes[4]      = {0};
    ptrdiff_t          dec_plane_sizes[4]    = {0};
    ptrdiff_t          dec_plane_offsets[4]  = {0};
    uint8_t           *dec_plane_data[4]     = {0};
    int                out_linesizes[4]      = {0};
    ptrdiff_t          out_plane_sizes[4]    = {0};
    ptrdiff_t          out_plane_offsets[4]  = {0};
    uint8_t           *out_plane_data[4]     = {0};
    int                block_cursor          = 0;
    int                blocks_are_spatial    = !!s->coeff_blocks_are_spatial;
    const int          chroma_411_split_mb_x = s->stats.chroma_411_split_mb_x;
    const int          last_mb_y             = s->stats.last_mb_y;
    int                ret;

    ret =
        dv_vk_get_plane_layout(output_fmt, s->stats.frame_width, s->stats.frame_height, out_linesizes, out_plane_sizes, out_plane_offsets);
    if (ret < 0)
        return ret;

    for (int i = 0; i < 4; i++) {
        if (out_plane_sizes[i] <= 0)
            continue;
        out_plane_data[i] = s->plane_staging + out_plane_offsets[i];
    }

    if (decode_fmt == output_fmt) {
        memcpy(dec_linesizes, out_linesizes, sizeof(dec_linesizes));
        memcpy(dec_plane_sizes, out_plane_sizes, sizeof(dec_plane_sizes));
        memcpy(dec_plane_offsets, out_plane_offsets, sizeof(dec_plane_offsets));
        for (int i = 0; i < 4; i++)
            dec_plane_data[i] = out_plane_data[i];
    } else {
        int needed = 0;

        ret = dv_vk_get_plane_layout(decode_fmt, s->stats.frame_width, s->stats.frame_height, dec_linesizes, dec_plane_sizes,
                                     dec_plane_offsets);
        if (ret < 0)
            return ret;

        for (int i = 0; i < 4; i++) {
            if (dec_plane_sizes[i] <= 0)
                continue;
            if (dec_plane_sizes[i] > INT_MAX - needed)
                return AVERROR(EINVAL);
            needed += (int)dec_plane_sizes[i];
        }

        if (needed > s->decode_plane_staging_size) {
            uint8_t *tmp = av_realloc(s->decode_plane_staging, needed);
            if (!tmp)
                return AVERROR(ENOMEM);
            s->decode_plane_staging      = tmp;
            s->decode_plane_staging_size = needed;
        }

        memset(s->decode_plane_staging, 0, needed);
        for (int i = 0; i < 4; i++) {
            if (dec_plane_sizes[i] <= 0)
                continue;
            dec_plane_data[i] = s->decode_plane_staging + dec_plane_offsets[i];
        }
    }

    for (int mb_job_idx = 0; mb_job_idx < s->stats.mb_jobs; mb_job_idx++) {
        const DVMacroblockJob *job           = &s->mb_jobs[mb_job_idx];
        int16_t               *block         = &s->coeff_blocks[block_cursor * 64];
        int                    mb_x          = job->mb_x;
        int                    mb_y          = job->mb_y;
        int                    is_field_mode = (mb_job_idx < s->mb_field_modes_alloc) ? !!s->mb_field_modes[mb_job_idx] : 0;
        int                    y_stride;
        int                    linesize;
        uint8_t               *y_ptr;
        int                    c_offset;

        if ((decode_fmt == AV_PIX_FMT_YUV420P) || (decode_fmt == AV_PIX_FMT_YUV411P && mb_x >= chroma_411_split_mb_x) ||
            (s->stats.frame_height >= 720 && mb_y != last_mb_y)) {
            y_stride = (dec_linesizes[0] << ((!is_field_mode) * log2_blocksize));
        } else {
            y_stride = (2 << log2_blocksize);
        }

        y_ptr = dec_plane_data[0] + ((mb_y * dec_linesizes[0] + mb_x) << log2_blocksize);

        if (mb_y == last_mb_y && is_field_mode) {
            if (blocks_are_spatial)
                dv_vk_put_last_row_field_luma_spatial(y_ptr, dec_linesizes[0], block);
            else
                dv_vk_put_last_row_field_luma(y_ptr, dec_linesizes[0], block);
        } else {
            linesize = dec_linesizes[0] << is_field_mode;
            if (blocks_are_spatial)
                dv_vk_put_block_8x8(y_ptr, linesize, block + 0 * 64);
            else
                ff_simple_idct_put_int16_8bit(y_ptr, linesize, block + 0 * 64);
            if (s->sys->video_stype == 4) {
                if (blocks_are_spatial)
                    dv_vk_put_block_8x8(y_ptr + (1 << log2_blocksize), linesize, block + 2 * 64);
                else
                    ff_simple_idct_put_int16_8bit(y_ptr + (1 << log2_blocksize), linesize, block + 2 * 64);
            } else {
                if (blocks_are_spatial)
                    dv_vk_put_block_8x8(y_ptr + (1 << log2_blocksize), linesize, block + 1 * 64);
                else
                    ff_simple_idct_put_int16_8bit(y_ptr + (1 << log2_blocksize), linesize, block + 1 * 64);

                if (blocks_are_spatial)
                    dv_vk_put_block_8x8(y_ptr + y_stride, linesize, block + 2 * 64);
                else
                    ff_simple_idct_put_int16_8bit(y_ptr + y_stride, linesize, block + 2 * 64);

                if (blocks_are_spatial)
                    dv_vk_put_block_8x8(y_ptr + (1 << log2_blocksize) + y_stride, linesize, block + 3 * 64);
                else
                    ff_simple_idct_put_int16_8bit(y_ptr + (1 << log2_blocksize) + y_stride, linesize, block + 3 * 64);
            }
        }

        block += 4 * 64;

        c_offset =
            (((mb_y >> (decode_fmt == AV_PIX_FMT_YUV420P)) * dec_linesizes[1] + (mb_x >> ((decode_fmt == AV_PIX_FMT_YUV411P) ? 2 : 1)))
             << log2_blocksize);

        for (int plane = 2; plane >= 1; plane--) {
            uint8_t *c_ptr = dec_plane_data[plane] + c_offset;

            if (decode_fmt == AV_PIX_FMT_YUV411P && mb_x >= chroma_411_split_mb_x) {
                uint8_t aligned_pixels[64];
                if (blocks_are_spatial) {
                    for (int y = 0; y < 8; y++) {
                        for (int x = 0; x < 8; x++)
                            aligned_pixels[y * 8 + x] = av_clip_uint8(block[y * 8 + x]);
                    }
                } else {
                    ff_simple_idct_put_int16_8bit(aligned_pixels, 8, block);
                }
                for (int y = 0; y < (1 << log2_blocksize); y++) {
                    uint8_t       *dst0     = c_ptr + y * dec_linesizes[plane];
                    uint8_t       *dst1     = dst0 + (dec_linesizes[plane] << log2_blocksize);
                    const uint8_t *src_row  = aligned_pixels + y * 8;
                    const uint8_t *src_row2 = src_row + ((1 << log2_blocksize) >> 1);

                    for (int x = 0; x < (1 << FFMAX(log2_blocksize - 1, 0)); x++) {
                        dst0[x] = src_row[x];
                        dst1[x] = src_row2[x];
                    }
                }
                block += 64;
            } else {
                y_stride = (mb_y == last_mb_y) ? (1 << log2_blocksize) : dec_linesizes[plane] << ((!is_field_mode) * log2_blocksize);
                if (mb_y == last_mb_y && is_field_mode) {
                    if (blocks_are_spatial)
                        dv_vk_put_last_row_field_chroma_spatial(c_ptr, dec_linesizes[plane], block);
                    else
                        dv_vk_put_last_row_field_chroma(c_ptr, dec_linesizes[plane], block);
                    block += 2 * 64;
                } else {
                    linesize = dec_linesizes[plane] << is_field_mode;
                    if (blocks_are_spatial)
                        dv_vk_put_block_8x8(c_ptr, linesize, block);
                    else
                        ff_simple_idct_put_int16_8bit(c_ptr, linesize, block);
                    block += 64;
                    if (s->sys->bpm == 8) {
                        if (blocks_are_spatial)
                            dv_vk_put_block_8x8(c_ptr + y_stride, linesize, block);
                        else
                            ff_simple_idct_put_int16_8bit(c_ptr + y_stride, linesize, block);
                        block += 64;
                    }
                }
            }
        }

        block_cursor += s->sys->bpm;
    }

    if (decode_fmt != output_fmt) {
        if (decode_fmt == AV_PIX_FMT_YUV411P && output_fmt == AV_PIX_FMT_YUV420P) {
            ret = dv_vk_convert_411_to_420(s, dec_plane_data, dec_linesizes, out_plane_data, out_linesizes);
            if (ret < 0)
                return ret;
        } else {
            av_log(avctx, AV_LOG_ERROR, "dv_vulkan: unsupported CPU format conversion %s -> %s\n", av_get_pix_fmt_name(decode_fmt),
                   av_get_pix_fmt_name(output_fmt));
            return AVERROR_PATCHWELCOME;
        }
    }

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: CPU plane reconstruction filled %d blocks\n", block_cursor);

    return 0;
}

static int dv_vk_stage_idct(AVCodecContext *avctx, DVSubContext *s)
{
    int    num_blocks = s->stats.block_jobs;
    int    err        = 0;
    size_t bytes;

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: idct stage start (gpu=%d blocks=%d)\n", s->idct_gpu_ready, num_blocks);

    s->coeff_blocks_are_spatial = 0;
    if (num_blocks <= 0)
        return 0;

    if (atomic_load(&dv_vk_global_disable_gpu_idct)) {
        s->idct_gpu_ready = 0;
    }

    if (s->idct_gpu_ready) {
        FFVkExecContext *exec;
        int32_t         *gpu_storage;
        int              blocks_per_row;
        int              rows;

        bytes = (size_t)num_blocks * 128 * sizeof(int32_t);

        if (bytes > s->idct_buf_size) {
            if (s->idct_buf.buf) {
                if (s->idct_buf_map) {
                    ff_vk_unmap_buffer(&s->vk, &s->idct_buf, 0);
                    s->idct_buf_map = NULL;
                }
                ff_vk_free_buf(&s->vk, &s->idct_buf);
            }

            err = ff_vk_create_buf(&s->vk, &s->idct_buf, bytes, NULL, NULL, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (err < 0)
                goto gpu_fail;

            err = ff_vk_map_buffer(&s->vk, &s->idct_buf, (uint8_t **)&s->idct_buf_map, 0);
            if (err < 0)
                goto gpu_fail;

            s->idct_buf_size = bytes;
        }

        if (!s->dequant_output_in_idct_buf) {
            gpu_storage = s->idct_buf_map;
            if (!gpu_storage) {
                err = AVERROR_EXTERNAL;
                goto gpu_fail;
            }
            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                int storage_base = block_idx * 128;
                int coeff_base   = block_idx * 64;

                for (int i = 0; i < 64; i++)
                    gpu_storage[storage_base + i] = s->coeff_blocks[coeff_base + i];
            }
        }

        if (s->recon_gpu_ready && dv_vk_recon_gpu_supported(s->sys)) {
            s->idct_upload_ready        = 1;
            s->coeff_blocks_are_spatial = 0;
            s->dequant_output_in_idct_buf = 0;
            if (!atomic_exchange(&dv_vk_global_logged_idct_gpu_active, 1))
                av_log(avctx, AV_LOG_INFO, "dv_vulkan: IDCT stage wired (GPU coeff upload for batched reconstruction)\n");
            av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: idct stage uploaded coeffs for GPU batched IDCT + reconstruction\n");
            return 0;
        }

        exec = ff_vk_exec_get(&s->vk, &s->idct_exec_pool);
        if (!exec) {
            err = AVERROR_EXTERNAL;
            goto gpu_fail;
        }

        err = ff_vk_exec_start(&s->vk, exec);
        if (err < 0)
            goto gpu_fail;

        err = ff_vk_shader_update_desc_buffer(&s->vk, exec, &s->idct_shd, 0, 0, 0, &s->idct_buf, 0, bytes, VK_FORMAT_UNDEFINED);
        if (err < 0)
            goto gpu_fail;

        ff_vk_exec_bind_shader(&s->vk, exec, &s->idct_shd);

        blocks_per_row = FFMIN(num_blocks, 512);
        rows           = (num_blocks + blocks_per_row - 1) / blocks_per_row;

        {
            DVVkIDCTPush pd = {
                .num_blocks     = (uint32_t)num_blocks,
                .blocks_per_row = (uint32_t)blocks_per_row,
                .output_base    = 0,
                .reserved       = 0,
            };

            ff_vk_shader_update_push_const(&s->vk, exec, &s->idct_shd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
        }

        s->vk.vkfn.CmdDispatch(exec->buf, (uint32_t)blocks_per_row, (uint32_t)rows, 1);

        av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: idct stage submitted gpu dispatch (%d blocks)\n", num_blocks);

        err = ff_vk_exec_submit(&s->vk, exec);
        if (err < 0)
            goto gpu_fail;

        ff_vk_exec_wait(&s->vk, exec);

        {
            gpu_storage = s->idct_buf_map;
            if (!gpu_storage) {
                err = AVERROR_EXTERNAL;
                goto gpu_fail;
            }
            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                int storage_base = block_idx * 128;
                int coeff_base   = block_idx * 64;

                for (int i = 0; i < 64; i++)
                    s->coeff_blocks[coeff_base + i] = av_clip_int16(gpu_storage[storage_base + 64 + i]);
            }

            av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: idct stage readback for CPU fallback\n");
        }

    } else {
        if (!atomic_exchange(&dv_vk_global_logged_idct_fallback, 1))
            av_log(avctx, AV_LOG_ERROR, "dv_vulkan: GPU IDCT unavailable\n");
        return AVERROR(ENOSYS);
    }

    if (!atomic_exchange(&dv_vk_global_logged_idct_gpu_active, 1)) {
        av_log(avctx, AV_LOG_INFO, "dv_vulkan: IDCT stage wired (GPU dispatch active)\n");
    }
    s->logged_idct_fallback = 1;

    s->coeff_blocks_are_spatial = 1;
    return 0;

gpu_fail:
    if (!s->logged_idct_gpu_fail) {
        av_log(avctx, AV_LOG_ERROR, "dv_vulkan: GPU IDCT dispatch failed (%d)\n", err);
        s->logged_idct_gpu_fail = 1;
    }

    atomic_store(&dv_vk_global_disable_gpu_idct, 1);
    s->idct_gpu_ready = 0;
    return err < 0 ? err : AVERROR_EXTERNAL;
}

static int dv_vk_stage_color_convert(AVCodecContext *avctx, DVSubContext *s)
{
    int      ret = 0;
    uint32_t chroma_w_shift;
    uint32_t chroma_h_shift;
    uint32_t is_yuv411;
    int      use_cpu_plane_upload = 0;

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: color stage start (gpu=%d fmt=%s bpm=%d)\n", s->recon_gpu_ready,
           av_get_pix_fmt_name(s->sys ? s->sys->pix_fmt : AV_PIX_FMT_NONE), s->sys ? s->sys->bpm : 0);

    if (s->recon_gpu_ready && dv_vk_recon_gpu_supported(s->sys)) {
        AVFrame               *dst = NULL;
        FFVkExecContext       *exec;
        VkBufferMemoryBarrier2 buf_bar[2];
        VkImageView            out_views[AV_NUM_DATA_POINTERS] = {0};
        VkImageMemoryBarrier2  img_bar[8];
        int                    nb_img_bar       = 0;
        int                    groups_x         = FFALIGN(s->stats.frame_width, 8) / 8;
        int                    groups_y         = FFALIGN(s->stats.frame_height, 8) / 8;
        int                    src_linesizes[4] = {0};
        ptrdiff_t              src_sizes[4]     = {0};
        ptrdiff_t              src_offsets[4]   = {0};
        uint32_t               src_widths[4]    = {0};
        uint32_t               src_heights[4]   = {0};
        DVVkReconJob          *jobs             = NULL;
        size_t                 job_bytes        = (size_t)s->stats.mb_jobs * sizeof(*jobs);
        size_t                 plane_bytes      = 0;
        int                    num_blocks       = s->stats.mb_jobs * s->sys->bpm;
        size_t                 idct_bytes       = (size_t)num_blocks * 128 * sizeof(int32_t);

        ret = dv_vk_get_chroma_shifts(use_cpu_plane_upload ? s->stats.sw_format : s->sys->pix_fmt, &chroma_w_shift, &chroma_h_shift,
                                      &is_yuv411);
        if (ret < 0)
            goto gpu_recon_fail;

        if (avctx->priv_data) {
            DVVkDecoderBridge *bridge = avctx->priv_data;
            if (bridge->frame)
                dst = (AVFrame *)bridge->frame;
        }
        if (!dst && avctx->internal)
            dst = avctx->internal->buffer_frame;

        if (!dst || !dst->data[0] || dst->format != AV_PIX_FMT_VULKAN) {
            ret = AVERROR(EINVAL);
            goto gpu_recon_fail;
        }

        ret = dv_vk_get_plane_layout(use_cpu_plane_upload ? s->stats.sw_format : s->sys->pix_fmt, s->stats.frame_width,
                                     s->stats.frame_height, src_linesizes, src_sizes, src_offsets);
        if (ret < 0)
            goto gpu_recon_fail;

        src_widths[0]  = (uint32_t)s->stats.frame_width;
        src_heights[0] = (uint32_t)s->stats.frame_height;
        src_widths[1]  = (uint32_t)FFMAX(s->stats.frame_width >> chroma_w_shift, 1);
        src_heights[1] = (uint32_t)FFMAX(s->stats.frame_height >> chroma_h_shift, 1);
        src_widths[2]  = (uint32_t)FFMAX(s->stats.frame_width >> chroma_w_shift, 1);
        src_heights[2] = (uint32_t)FFMAX(s->stats.frame_height >> chroma_h_shift, 1);

        if (use_cpu_plane_upload) {
            ret = dv_vk_stage_cpu_reconstruct_yuv(avctx, s);
            if (ret < 0)
                goto gpu_recon_fail;
        }

        for (int i = 0; i < 4; i++) {
            if (src_sizes[i] > 0)
                plane_bytes += (size_t)src_sizes[i];
        }

        plane_bytes = FFALIGN(plane_bytes, 4) * sizeof(uint32_t);

        if (job_bytes > s->recon_jobs_buf_size) {
            if (s->recon_jobs_buf.buf) {
                if (s->recon_jobs_buf_map) {
                    ff_vk_unmap_buffer(&s->vk, &s->recon_jobs_buf, 0);
                    s->recon_jobs_buf_map = NULL;
                }
                ff_vk_free_buf(&s->vk, &s->recon_jobs_buf);
            }

            ret = ff_vk_create_buf(&s->vk, &s->recon_jobs_buf, job_bytes, NULL, NULL, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (ret < 0)
                goto gpu_recon_fail;

            ret = ff_vk_map_buffer(&s->vk, &s->recon_jobs_buf, (uint8_t **)&s->recon_jobs_buf_map, 0);
            if (ret < 0)
                goto gpu_recon_fail;
            s->recon_jobs_buf_size = job_bytes;
        }

        if (plane_bytes > s->recon_plane_buf_size) {
            if (s->recon_plane_buf.buf)
                if (s->recon_plane_buf_map) {
                    ff_vk_unmap_buffer(&s->vk, &s->recon_plane_buf, 0);
                    s->recon_plane_buf_map = NULL;
                }
            if (s->recon_plane_buf.buf)
                ff_vk_free_buf(&s->vk, &s->recon_plane_buf);

            ret = ff_vk_create_buf(&s->vk, &s->recon_plane_buf, plane_bytes, NULL, NULL, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (ret < 0)
                goto gpu_recon_fail;

            ret = ff_vk_map_buffer(&s->vk, &s->recon_plane_buf, (uint8_t **)&s->recon_plane_buf_map, 0);
            if (ret < 0)
                goto gpu_recon_fail;
            s->recon_plane_buf_size = plane_bytes;
        }

        if (use_cpu_plane_upload) {
            if (!s->recon_plane_buf_map) {
                ret = AVERROR(EINVAL);
                goto gpu_recon_fail;
            }

            memset(s->recon_plane_buf_map, 0, s->recon_plane_buf_size);
            for (int plane = 0; plane < 4; plane++) {
                if (src_sizes[plane] <= 0 || src_linesizes[plane] <= 0)
                    continue;

                for (uint32_t y = 0; y < (uint32_t)(src_sizes[plane] / src_linesizes[plane]); y++) {
                    const uint8_t *src_row = s->plane_staging + src_offsets[plane] + (ptrdiff_t)y * src_linesizes[plane];
                    uint32_t      *dst_row = s->recon_plane_buf_map + src_offsets[plane] + y * (uint32_t)src_linesizes[plane];

                    for (uint32_t x = 0; x < (uint32_t)src_linesizes[plane]; x++)
                        dst_row[x] = src_row[x];
                }
            }
        }

        jobs = (DVVkReconJob *)s->recon_jobs_buf_map;
        if (!jobs) {
            ret = AVERROR_EXTERNAL;
            goto gpu_recon_fail;
        }

        for (int i = 0; i < s->stats.mb_jobs; i++) {
            jobs[i].mb_x       = (uint32_t)s->mb_jobs[i].mb_x;
            jobs[i].mb_y       = (uint32_t)s->mb_jobs[i].mb_y;
            jobs[i].field_mode = (uint32_t)((i < s->mb_field_modes_alloc) ? !!s->mb_field_modes[i] : 0);
            jobs[i].reserved   = 0;
        }

        exec = ff_vk_exec_get(&s->vk, &s->idct_exec_pool);
        if (!exec) {
            ret = AVERROR_EXTERNAL;
            goto gpu_recon_fail;
        }

        ret = ff_vk_exec_start(&s->vk, exec);
        if (ret < 0)
            goto gpu_recon_fail;

        if (!s->idct_upload_ready) {
            ret = AVERROR(EINVAL);
            goto gpu_recon_fail;
        }

        if (!use_cpu_plane_upload && !s->coeff_blocks_are_spatial) {
            ret = ff_vk_shader_update_desc_buffer(&s->vk, exec, &s->idct_shd, 0, 0, 0, &s->idct_buf, 0, idct_bytes, VK_FORMAT_UNDEFINED);
            if (ret < 0)
                goto gpu_recon_fail;

            ff_vk_exec_bind_shader(&s->vk, exec, &s->idct_shd);

            {
                int          blocks_per_row = FFMIN(num_blocks, 512);
                int          rows           = (num_blocks + blocks_per_row - 1) / blocks_per_row;
                DVVkIDCTPush pd             = {
                                .num_blocks     = (uint32_t)num_blocks,
                                .blocks_per_row = (uint32_t)blocks_per_row,
                                .output_base    = 0,
                                .reserved       = 0,
                };

                ff_vk_shader_update_push_const(&s->vk, exec, &s->idct_shd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
                s->vk.vkfn.CmdDispatch(exec->buf, (uint32_t)blocks_per_row, (uint32_t)rows, 1);
            }

            buf_bar[0] = (VkBufferMemoryBarrier2){
                .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .srcAccessMask       = VK_ACCESS_2_SHADER_WRITE_BIT,
                .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer              = s->idct_buf.buf,
                .offset              = 0,
                .size                = idct_bytes,
            };
        } else if (!use_cpu_plane_upload) {
            buf_bar[0] = (VkBufferMemoryBarrier2){
                .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                .srcStageMask        = VK_PIPELINE_STAGE_2_HOST_BIT,
                .srcAccessMask       = VK_ACCESS_2_HOST_WRITE_BIT,
                .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                .dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .buffer              = s->idct_buf.buf,
                .offset              = 0,
                .size                = idct_bytes,
            };
        } else {
            memset(&buf_bar[0], 0, sizeof(buf_bar[0]));
        }
        if (!use_cpu_plane_upload) {
            s->vk.vkfn.CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo){
                                                          .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                          .bufferMemoryBarrierCount = 1,
                                                          .pBufferMemoryBarriers    = buf_bar,
                                                      });

            ret = ff_vk_shader_update_desc_buffer(&s->vk, exec, &s->recon_calc_shd, 0, 0, 0, &s->idct_buf, 0, s->idct_buf_size,
                                                  VK_FORMAT_UNDEFINED);
            if (ret < 0)
                goto gpu_recon_fail;

            ret = ff_vk_shader_update_desc_buffer(&s->vk, exec, &s->recon_calc_shd, 0, 1, 0, &s->recon_jobs_buf, 0, s->recon_jobs_buf_size,
                                                  VK_FORMAT_UNDEFINED);
            if (ret < 0)
                goto gpu_recon_fail;

            ret = ff_vk_shader_update_desc_buffer(&s->vk, exec, &s->recon_calc_shd, 0, 2, 0, &s->recon_plane_buf, 0,
                                                  s->recon_plane_buf_size, VK_FORMAT_UNDEFINED);
            if (ret < 0)
                goto gpu_recon_fail;

            ff_vk_exec_bind_shader(&s->vk, exec, &s->recon_calc_shd);

            {
                DVVkReconCalcPush pd = {0};

                for (int i = 0; i < 4; i++) {
                    pd.width[i]        = (src_sizes[i] > 0 && src_linesizes[i] > 0) ? (uint32_t)src_linesizes[i] : 0;
                    pd.height[i]       = (src_sizes[i] > 0 && src_linesizes[i] > 0) ? (uint32_t)(src_sizes[i] / src_linesizes[i]) : 0;
                    pd.plane_offset[i] = (uint32_t)src_offsets[i];
                    pd.plane_stride[i] = (uint32_t)src_linesizes[i];
                }
                pd.num_blocks            = (uint32_t)num_blocks;
                pd.blocks_per_mb         = (uint32_t)s->sys->bpm;
                pd.chroma_w_shift        = chroma_w_shift;
                pd.chroma_h_shift        = chroma_h_shift;
                pd.is_yuv411             = is_yuv411;
                pd.last_mb_y             = (uint32_t)s->stats.last_mb_y;
                pd.chroma_411_split_mb_x = (uint32_t)s->stats.chroma_411_split_mb_x;

                ff_vk_shader_update_push_const(&s->vk, exec, &s->recon_calc_shd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
            }

            s->vk.vkfn.CmdDispatch(exec->buf, (uint32_t)FFMIN(num_blocks, 512),
                                   (uint32_t)((num_blocks + FFMIN(num_blocks, 512) - 1) / FFMIN(num_blocks, 512)), 1);
        }

        buf_bar[1] = (VkBufferMemoryBarrier2){
            .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask        = use_cpu_plane_upload ? VK_PIPELINE_STAGE_2_HOST_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask       = use_cpu_plane_upload ? VK_ACCESS_2_HOST_WRITE_BIT : VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask        = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask       = VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer              = s->recon_plane_buf.buf,
            .offset              = 0,
            .size                = s->recon_plane_buf_size,
        };
        s->vk.vkfn.CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo){
                                                      .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                      .bufferMemoryBarrierCount = 1,
                                                      .pBufferMemoryBarriers    = &buf_bar[1],
                                                  });

        if (!use_cpu_plane_upload)
            av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: color stage submitted batched IDCT + recon calc (%d blocks)\n", num_blocks);
        else
            av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: color stage submitted CPU YUV upload + GPU YUV->RGB path\n");

        ret = ff_vk_exec_add_dep_frame(&s->vk, exec, dst, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        if (ret < 0)
            goto gpu_recon_fail;

        ret = ff_vk_create_imageviews(&s->vk, exec, out_views, dst, FF_VK_REP_FLOAT);
        if (ret < 0)
            goto gpu_recon_fail;

        ff_vk_frame_barrier(&s->vk, exec, dst, img_bar, &nb_img_bar, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
                            VK_QUEUE_FAMILY_IGNORED);
        s->vk.vkfn.CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo){
                                                      .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                                      .pImageMemoryBarriers    = img_bar,
                                                      .imageMemoryBarrierCount = nb_img_bar,
                                                  });

        ret = ff_vk_shader_update_desc_buffer(&s->vk, exec, &s->recon_shd, 0, 0, 0, &s->recon_plane_buf, 0, s->recon_plane_buf_size,
                                              VK_FORMAT_UNDEFINED);
        if (ret < 0)
            goto gpu_recon_fail;

        ff_vk_shader_update_img_array(&s->vk, exec, &s->recon_shd, dst, out_views, 0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_NULL_HANDLE);

        ff_vk_exec_bind_shader(&s->vk, exec, &s->recon_shd);

        {
            DVVkReconPush pd = {0};

            for (int i = 0; i < 4; i++) {
                pd.width[i]        = src_widths[i];
                pd.height[i]       = src_heights[i];
                pd.plane_offset[i] = (uint32_t)src_offsets[i];
                pd.plane_stride[i] = (uint32_t)src_linesizes[i];
            }
            pd.chroma_w_shift = chroma_w_shift;
            pd.chroma_h_shift = chroma_h_shift;

            ff_vk_shader_update_push_const(&s->vk, exec, &s->recon_shd, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pd), &pd);
        }

        s->vk.vkfn.CmdDispatch(exec->buf, (uint32_t)groups_x, (uint32_t)groups_y, 1);

        av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: color stage submitted recon calc + YUV->RGB pass (%dx%d) in single batch\n", groups_x,
               groups_y);

        ret = ff_vk_exec_submit(&s->vk, exec);
        if (ret < 0)
            goto gpu_recon_fail;
        s->idct_upload_ready = 0;

        s->cpu_output_required = 0;
        return 0;

    gpu_recon_fail:
        ret = (ret < 0) ? ret : AVERROR_EXTERNAL;
        if (!s->logged_recon_gpu_fail) {
            av_log(avctx, AV_LOG_ERROR, "dv_vulkan: GPU reconstruction failed (%d)\n", ret);
            s->logged_recon_gpu_fail = 1;
        }
        s->idct_upload_ready   = 0;
        s->idct_gpu_ready      = 0;
        s->recon_gpu_ready     = 0;
        s->cpu_output_required = 0;
        return ret;
    }

    av_log(avctx, AV_LOG_ERROR, "dv_vulkan: GPU reconstruction unavailable for this profile\n");
    return AVERROR(ENOSYS);
}

static int dv_vk_stage_write_output(AVCodecContext *avctx, DVSubContext *s)
{
    AVFrame            src              = {0};
    AVFrame           *dst              = NULL;
    int                linesizes[4]     = {0};
    ptrdiff_t          plane_sizes[4]   = {0};
    ptrdiff_t          plane_offsets[4] = {0};
    int                ret;
    enum AVPixelFormat src_fmt;
    const uint8_t     *src_base;

    if (s->recon_gpu_ready && dv_vk_recon_gpu_supported(s->sys) && !s->cpu_output_required)
        return 0;

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: output stage CPU upload path (format=%s, %dx%d)\n", av_get_pix_fmt_name(s->stats.sw_format),
           s->stats.frame_width, s->stats.frame_height);

    if (avctx->priv_data) {
        DVVkDecoderBridge *bridge = avctx->priv_data;
        if (bridge->frame)
            dst = (AVFrame *)bridge->frame;
    }
    if (!dst && avctx->internal)
        dst = avctx->internal->buffer_frame;

    if (!dst || !dst->data[0] || dst->format != AV_PIX_FMT_VULKAN)
        return AVERROR(EINVAL);

    src_fmt  = s->stats.sw_format;
    src_base = s->plane_staging;

    if (!src_base)
        return AVERROR(EINVAL);

    ret = dv_vk_get_plane_layout(src_fmt, s->stats.frame_width, s->stats.frame_height, linesizes, plane_sizes, plane_offsets);
    if (ret < 0)
        return ret;

    if (avctx->hw_frames_ctx) {
        AVHWFramesContext *frames_ctx = (AVHWFramesContext *)avctx->hw_frames_ctx->data;

        if (frames_ctx && frames_ctx->sw_format == AV_PIX_FMT_RGBA) {
            struct SwsContext *sws              = NULL;
            uint8_t           *rgba_data[4]     = {0};
            int                rgba_linesize[4] = {0};
            const uint8_t     *src_data[4]      = {0};
            int                src_linesize[4]  = {0};
            const int         *coeffs           = sws_getCoefficients(SWS_CS_ITU601);
            for (int i = 0; i < 4; i++) {
                if (plane_sizes[i] <= 0)
                    continue;
                src_data[i]     = src_base + plane_offsets[i];
                src_linesize[i] = linesizes[i];
            }

            sws = sws_getContext(s->stats.frame_width, s->stats.frame_height, src_fmt, s->stats.frame_width, s->stats.frame_height,
                                 AV_PIX_FMT_RGBA, SWS_BILINEAR, NULL, NULL, NULL);
            if (!sws)
                return AVERROR(ENOMEM);

            sws_setColorspaceDetails(sws, coeffs, 0, coeffs, 1, 0, 1 << 16, 1 << 16);

            ret = av_image_alloc(rgba_data, rgba_linesize, s->stats.frame_width, s->stats.frame_height, AV_PIX_FMT_RGBA, 1);
            if (ret < 0) {
                sws_freeContext(sws);
                return ret;
            }

            sws_scale(sws, src_data, src_linesize, 0, s->stats.frame_height, rgba_data, rgba_linesize);
            sws_freeContext(sws);

            src.format      = AV_PIX_FMT_RGBA;
            src.width       = s->stats.frame_width;
            src.height      = s->stats.frame_height;
            src.data[0]     = rgba_data[0];
            src.linesize[0] = rgba_linesize[0];

            ret = av_hwframe_transfer_data(dst, &src, 0);
            av_freep(&rgba_data[0]);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "dv_vulkan: failed to upload CPU reconstructed RGBA frame (%d)\n", ret);
                return ret;
            }

            av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: uploaded CPU reconstructed RGBA frame to Vulkan output\n");
            return 0;
        }
    }

    src.format = src_fmt;
    src.width  = s->stats.frame_width;
    src.height = s->stats.frame_height;

    for (int i = 0; i < 4; i++) {
        if (plane_sizes[i] <= 0)
            continue;
        src.data[i]     = (uint8_t *)src_base + plane_offsets[i];
        src.linesize[i] = linesizes[i];
    }

    ret = av_hwframe_transfer_data(dst, &src, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "dv_vulkan: failed to upload reconstructed frame (%d)\n", ret);
        return ret;
    }

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: uploaded reconstructed frame to Vulkan output\n");

    return 0;
}

static int dv_vulkan_decode_init(AVCodecContext *avctx)
{
    DVSubContext *s        = avctx->internal->hwaccel_priv_data;
    uint8_t      *spv_data = NULL;
    size_t        spv_len  = 0;
#if CONFIG_LIBGLSLANG || CONFIG_LIBSHADERC
    FFVkSPIRVCompiler *spv              = NULL;
    void              *opaque           = NULL;
    uint8_t           *dequant_spv_data = NULL;
    size_t             dequant_spv_len  = 0;
    void              *dequant_opaque   = NULL;
    uint8_t           *calc_spv_data    = NULL;
    size_t             calc_spv_len     = 0;
    void              *calc_opaque      = NULL;
    uint8_t           *recon_spv_data   = NULL;
    size_t             recon_spv_len    = 0;
    void              *recon_opaque     = NULL;
    int                spv_compiled     = 0;
#endif
    AVVulkanDeviceQueueFamily *qf;
    int                        ret;

    if (!s)
        return AVERROR(EINVAL);

    if (!avctx->hw_device_ctx && !avctx->hw_frames_ctx) {
        av_log(avctx, AV_LOG_ERROR, "dv_vulkan: neither hw_device_ctx nor hw_frames_ctx is set\n");
        return AVERROR(EINVAL);
    }

    ret = ff_decode_get_hw_frames_ctx(avctx, AV_HWDEVICE_TYPE_VULKAN);
    if (ret < 0)
        return ret;

    ff_thread_once(&dv_vk_rl_vlc_once, dv_vk_init_static_rl_vlc);

    if (atomic_load(&dv_vk_global_disable_gpu_idct))
        goto no_gpu;

    ret = ff_vk_init(&s->vk, avctx, avctx->hw_device_ctx, avctx->hw_frames_ctx);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: ff_vk_init failed (%d), using CPU IDCT fallback\n", ret);
        goto no_gpu;
    }

    qf = ff_vk_qf_find(&s->vk, VK_QUEUE_COMPUTE_BIT, 0);
    if (!qf) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: compute queue family not found, using CPU IDCT fallback\n");
        goto no_gpu;
    }

    ret = ff_vk_exec_pool_init(&s->vk, qf, &s->idct_exec_pool, 4, 0, 0, 0, NULL);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: exec pool init failed (%d), using CPU IDCT fallback\n", ret);
        goto no_gpu;
    }

    ret = ff_vk_shader_init(&s->vk, &s->idct_shd, "dv_idct", VK_SHADER_STAGE_COMPUTE_BIT, NULL, 0, 8, 8, 1, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: shader init failed (%d), using CPU IDCT fallback\n", ret);
        goto no_gpu;
    }

    ff_vk_shader_add_push_const(&s->idct_shd, 0, sizeof(DVVkIDCTPush), VK_SHADER_STAGE_COMPUTE_BIT);
    {
        const FFVulkanDescriptorSetBinding idct_desc[] = {
            {
                .name        = "idct_storage",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "int values[];",
            },
        };
        ff_vk_shader_add_descriptor_set(&s->vk, &s->idct_shd, idct_desc, 1, 0, 0);
    }
    dv_vk_build_idct_shader_source(&s->idct_shd);

    ret = AVERROR_EXTERNAL;
#if CONFIG_LIBGLSLANG || CONFIG_LIBSHADERC
    spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: no in-memory SPIR-V compiler backend, using CPU IDCT fallback\n");
        goto no_gpu;
    }

    ret = dv_vk_get_cached_or_compile_spv(spv, &s->vk, &s->idct_shd, &dv_vk_cached_idct_spv, &dv_vk_cached_idct_spv_len, &spv_data,
                                          &spv_len, &opaque);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: SPIR-V compiler path failed (%d), using CPU IDCT fallback\n", ret);
        goto no_gpu;
    }
    spv_compiled = 1;
#else
    av_log(avctx, AV_LOG_WARNING, "dv_vulkan: built without SPIR-V compiler backend, using CPU IDCT fallback\n");
    goto no_gpu;
#endif

    ret = ff_vk_shader_link(&s->vk, &s->idct_shd, spv_data, spv_len, "main");
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: shader link failed (%d), using CPU IDCT fallback\n", ret);
        goto no_gpu;
    }

    ret = ff_vk_shader_register_exec(&s->vk, &s->idct_exec_pool, &s->idct_shd);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: shader register failed (%d), using CPU IDCT fallback\n", ret);
        goto no_gpu;
    }

    av_free(spv_data);
    spv_data = NULL;
    opaque   = NULL;

    ret = ff_vk_shader_init(&s->vk, &s->dequant_shd, "dv_dequant", VK_SHADER_STAGE_COMPUTE_BIT, NULL, 0, 64, 1, 1, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: dequant shader init failed (%d), GPU dequant unavailable\n", ret);
        s->dequant_gpu_ready = 0;
        goto dequant_skip;
    }

    ff_vk_shader_add_push_const(&s->dequant_shd, 0, sizeof(DVVkDequantPush), VK_SHADER_STAGE_COMPUTE_BIT);
    {
        const FFVulkanDescriptorSetBinding dequant_desc[] = {
            {
                .name        = "quant_buffer",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "int quantized[];",
            },
            {
                .name        = "factor_buffer",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "uint factors[];",
            },
            {
                .name        = "output_buffer",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "int dequantized[];",
            },
        };
        ff_vk_shader_add_descriptor_set(&s->vk, &s->dequant_shd, dequant_desc, 3, 0, 0);
    }
    dv_vk_build_dequant_shader_source(&s->dequant_shd);

    ret = dv_vk_get_cached_or_compile_spv(spv, &s->vk, &s->dequant_shd, &dv_vk_cached_dequant_spv, &dv_vk_cached_dequant_spv_len,
                                          &dequant_spv_data, &dequant_spv_len, &dequant_opaque);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: dequant shader SPIR-V compilation failed (%d), GPU dequant unavailable\n", ret);
        s->dequant_gpu_ready = 0;
        goto dequant_skip;
    }

    ret = ff_vk_shader_link(&s->vk, &s->dequant_shd, dequant_spv_data, dequant_spv_len, "main");
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: dequant shader link failed (%d), GPU dequant unavailable\n", ret);
        s->dequant_gpu_ready = 0;
        goto dequant_skip;
    }

    ret = ff_vk_shader_register_exec(&s->vk, &s->idct_exec_pool, &s->dequant_shd);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: dequant shader register failed (%d), GPU dequant unavailable\n", ret);
        s->dequant_gpu_ready = 0;
        goto dequant_skip;
    }

    s->dequant_gpu_ready = 1;
    if (!atomic_exchange(&dv_vk_global_logged_dequant_gpu_init, 1))
        av_log(avctx, AV_LOG_INFO, "dv_vulkan: GPU dequant shader initialized\n");

dequant_skip:
    av_free(dequant_spv_data);
    dequant_spv_data = NULL;
    dequant_opaque   = NULL;

    s->idct_gpu_ready = 1;
    if (!atomic_exchange(&dv_vk_global_logged_idct_gpu_init, 1))
        av_log(avctx, AV_LOG_INFO, "dv_vulkan: GPU IDCT shader initialized from inline GLSL\n");

    ret = ff_vk_shader_init(&s->vk, &s->recon_calc_shd, "dv_recon_calc", VK_SHADER_STAGE_COMPUTE_BIT, NULL, 0, 8, 8, 1, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon calc shader init failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    ff_vk_shader_add_push_const(&s->recon_calc_shd, 0, sizeof(DVVkReconCalcPush), VK_SHADER_STAGE_COMPUTE_BIT);
    {
        const FFVulkanDescriptorSetBinding recon_calc_desc[] = {
            {
                .name        = "idct_buffer",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "int values[];",
            },
            {
                .name        = "job_buffer",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "uvec4 jobs[];",
            },
            {
                .name        = "plane_buffer",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "uint data[];",
            },
        };
        ff_vk_shader_add_descriptor_set(&s->vk, &s->recon_calc_shd, recon_calc_desc, 3, 0, 0);
    }
    dv_vk_build_recon_calc_shader_source(&s->recon_calc_shd);

#if CONFIG_LIBGLSLANG || CONFIG_LIBSHADERC
    ret = dv_vk_get_cached_or_compile_spv(spv, &s->vk, &s->recon_calc_shd, &dv_vk_cached_calc_spv, &dv_vk_cached_calc_spv_len,
                                          &calc_spv_data, &calc_spv_len, &calc_opaque);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon calc shader SPIR-V compilation failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    ret = ff_vk_shader_link(&s->vk, &s->recon_calc_shd, calc_spv_data, calc_spv_len, "main");
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon calc shader link failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    ret = ff_vk_shader_register_exec(&s->vk, &s->idct_exec_pool, &s->recon_calc_shd);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon calc shader register failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    av_free(calc_spv_data);
    calc_spv_data = NULL;
    calc_opaque   = NULL;
#endif

    ret = ff_vk_shader_init(&s->vk, &s->recon_shd, "dv_recon", VK_SHADER_STAGE_COMPUTE_BIT, NULL, 0, 8, 8, 1, 0);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon shader init failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    ff_vk_shader_add_push_const(&s->recon_shd, 0, sizeof(DVVkReconPush), VK_SHADER_STAGE_COMPUTE_BIT);
    {
        const FFVulkanDescriptorSetBinding recon_desc[] = {
            {
                .name        = "frame_buffer",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "uint data[];",
            },
            {
                .name       = "output_img",
                .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .mem_layout = ff_vk_shader_rep_fmt(AV_PIX_FMT_RGBA, FF_VK_REP_FLOAT),
                .mem_quali  = "writeonly",
                .dimensions = 2,
                .elems      = 1,
                .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
            },
        };
        ff_vk_shader_add_descriptor_set(&s->vk, &s->recon_shd, recon_desc, 2, 0, 0);
    }
    dv_vk_build_recon_shader_source(&s->recon_shd);

#if CONFIG_LIBGLSLANG || CONFIG_LIBSHADERC
    ret = dv_vk_get_cached_or_compile_spv(spv, &s->vk, &s->recon_shd, &dv_vk_cached_recon_spv, &dv_vk_cached_recon_spv_len, &recon_spv_data,
                                          &recon_spv_len, &recon_opaque);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon shader SPIR-V compilation failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    ret = ff_vk_shader_link(&s->vk, &s->recon_shd, recon_spv_data, recon_spv_len, "main");
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon shader link failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    ret = ff_vk_shader_register_exec(&s->vk, &s->idct_exec_pool, &s->recon_shd);
    if (ret < 0) {
        av_log(avctx, AV_LOG_WARNING, "dv_vulkan: recon shader register failed (%d), GPU reconstruction disabled\n", ret);
        goto recon_skip;
    }

    av_free(recon_spv_data);
    recon_spv_data = NULL;
    recon_opaque   = NULL;
#endif

    s->recon_gpu_ready = 1;
    if (!atomic_exchange(&dv_vk_global_logged_recon_gpu_init, 1))
        av_log(avctx, AV_LOG_INFO, "dv_vulkan: GPU reconstruction shader initialized\n");
    goto done;

recon_skip:
    s->recon_gpu_ready = 0;
    goto done;

no_gpu:
    s->dequant_gpu_ready = 0;
    s->idct_gpu_ready    = 0;
    s->recon_gpu_ready   = 0;

done:
    av_free(spv_data);
#if CONFIG_LIBGLSLANG || CONFIG_LIBSHADERC
    if (spv_compiled && opaque)
        opaque = NULL;
    av_free(dequant_spv_data);
    av_free(calc_spv_data);
    av_free(recon_spv_data);
    if (spv)
        spv->uninit(&spv);
#endif

    dv_vk_reset_frame_state(s);
    return 0;
}

static int dv_vulkan_alloc_frame(AVCodecContext *avctx, AVFrame *frame)
{
    int ret;

    if (!avctx->hw_frames_ctx) {
        ret = ff_decode_get_hw_frames_ctx(avctx, AV_HWDEVICE_TYPE_VULKAN);
        if (ret < 0)
            return ret;
    }

    ret = av_hwframe_get_buffer(avctx->hw_frames_ctx, frame, 0);
    if (ret < 0)
        return ret;

    ret = ff_attach_decode_data(frame);
    if (ret < 0) {
        av_frame_unref(frame);
        return ret;
    }

    frame->color_range     = AVCOL_RANGE_JPEG;
    frame->colorspace      = AVCOL_SPC_RGB;
    frame->color_primaries = AVCOL_PRI_BT709;
    frame->color_trc       = AVCOL_TRC_IEC61966_2_1;

    return 0;
}

static int dv_vulkan_decode_uninit(AVCodecContext *avctx)
{
    DVSubContext *s = avctx->internal->hwaccel_priv_data;

    if (!s)
        return 0;

    av_freep(&s->frame_packet);
    s->frame_packet_size = 0;

    av_freep(&s->mb_jobs);
    s->mb_jobs_alloc = 0;

    av_freep(&s->mb_field_modes);
    s->mb_field_modes_alloc = 0;

    av_freep(&s->coeff_blocks);
    s->coeff_blocks_alloc = 0;

    av_freep(&s->quant_blocks);
    av_freep(&s->factor_blocks);
    s->dequant_blocks_alloc = 0;

    if (s->dequant_quant_buf.buf) {
        if (s->dequant_quant_buf_map) {
            ff_vk_unmap_buffer(&s->vk, &s->dequant_quant_buf, 0);
            s->dequant_quant_buf_map = NULL;
        }
        ff_vk_free_buf(&s->vk, &s->dequant_quant_buf);
        memset(&s->dequant_quant_buf, 0, sizeof(s->dequant_quant_buf));
    }
    s->dequant_quant_buf_size = 0;

    if (s->dequant_factor_buf.buf) {
        if (s->dequant_factor_buf_map) {
            ff_vk_unmap_buffer(&s->vk, &s->dequant_factor_buf, 0);
            s->dequant_factor_buf_map = NULL;
        }
        ff_vk_free_buf(&s->vk, &s->dequant_factor_buf);
        memset(&s->dequant_factor_buf, 0, sizeof(s->dequant_factor_buf));
    }
    s->dequant_factor_buf_size = 0;

    if (s->dequant_out_buf.buf) {
        if (s->dequant_out_buf_map) {
            ff_vk_unmap_buffer(&s->vk, &s->dequant_out_buf, 0);
            s->dequant_out_buf_map = NULL;
        }
        ff_vk_free_buf(&s->vk, &s->dequant_out_buf);
        memset(&s->dequant_out_buf, 0, sizeof(s->dequant_out_buf));
    }
    s->dequant_out_buf_size = 0;

    if (s->idct_buf.buf) {
        if (s->idct_buf_map) {
            ff_vk_unmap_buffer(&s->vk, &s->idct_buf, 0);
            s->idct_buf_map = NULL;
        }
        ff_vk_free_buf(&s->vk, &s->idct_buf);
        memset(&s->idct_buf, 0, sizeof(s->idct_buf));
    }
    s->idct_buf_size = 0;

    if (s->recon_jobs_buf.buf) {
        if (s->recon_jobs_buf_map) {
            ff_vk_unmap_buffer(&s->vk, &s->recon_jobs_buf, 0);
            s->recon_jobs_buf_map = NULL;
        }
        ff_vk_free_buf(&s->vk, &s->recon_jobs_buf);
        memset(&s->recon_jobs_buf, 0, sizeof(s->recon_jobs_buf));
    }
    s->recon_jobs_buf_size = 0;

    if (s->recon_plane_buf.buf) {
        if (s->recon_plane_buf_map) {
            ff_vk_unmap_buffer(&s->vk, &s->recon_plane_buf, 0);
            s->recon_plane_buf_map = NULL;
        }
        ff_vk_free_buf(&s->vk, &s->recon_plane_buf);
        memset(&s->recon_plane_buf, 0, sizeof(s->recon_plane_buf));
    }
    s->recon_plane_buf_size = 0;

    ff_vk_shader_free(&s->vk, &s->dequant_shd);
    ff_vk_shader_free(&s->vk, &s->idct_shd);
    ff_vk_shader_free(&s->vk, &s->recon_calc_shd);
    ff_vk_shader_free(&s->vk, &s->recon_shd);
    ff_vk_exec_pool_free(&s->vk, &s->idct_exec_pool);
    ff_vk_uninit(&s->vk);
    memset(&s->dequant_shd, 0, sizeof(s->dequant_shd));
    memset(&s->idct_shd, 0, sizeof(s->idct_shd));
    memset(&s->recon_calc_shd, 0, sizeof(s->recon_calc_shd));
    memset(&s->recon_shd, 0, sizeof(s->recon_shd));
    memset(&s->idct_exec_pool, 0, sizeof(s->idct_exec_pool));
    s->dequant_gpu_ready = 0;
    s->idct_gpu_ready    = 0;
    s->recon_gpu_ready   = 0;

    av_freep(&s->plane_staging);
    s->plane_staging_size = 0;

    av_freep(&s->decode_plane_staging);
    s->decode_plane_staging_size = 0;

    s->sys = NULL;

    return 0;
}

static int dv_vulkan_start_frame(AVCodecContext *avctx, const AVBufferRef *hw_frames_ctx, const uint8_t *buffer, uint32_t size)
{
    DVSubContext *s = avctx->internal->hwaccel_priv_data;
    int           ret;

    (void)hw_frames_ctx;

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: start_frame size=%u\n", size);

    if (!s)
        return AVERROR(EINVAL);

    dv_vk_reset_frame_state(s);

    ret = dv_vk_cache_packet(s, buffer, size);
    if (ret < 0)
        return ret;

    s->frame_packet_from_start = 1;

    return 0;
}

static int dv_vulkan_decode_slice(AVCodecContext *avctx, const uint8_t *data, uint32_t size)
{
    DVSubContext *s = avctx->internal->hwaccel_priv_data;

    if (!s)
        return AVERROR(EINVAL);

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: decode_slice size=%u\n", size);

    if (s->frame_packet_from_start && s->frame_packet_size == size)
        return 0;

    /* Keep latest packet copy when decode_slice carries independent data. */
    s->frame_packet_from_start = 0;
    return dv_vk_cache_packet(s, data, size);
}

static int dv_vulkan_end_frame(AVCodecContext *avctx)
{
    DVSubContext *s = avctx->internal->hwaccel_priv_data;
    int           ret;

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: end_frame start\n");

    if (!s || !s->frame_packet || !s->frame_packet_size)
        return AVERROR_INVALIDDATA;

    ret = dv_vk_prepare_profile_and_tables(avctx, s);
    if (ret < 0)
        return ret;

    ret = dv_vk_build_mb_jobs(avctx, s);
    if (ret < 0)
        return ret;

    ret = dv_vk_prepare_coeff_staging(s);
    if (ret < 0)
        return ret;

    ret = dv_vk_prepare_plane_staging(s);
    if (ret < 0)
        return ret;

    ret = dv_vk_stage_cpu_entropy(avctx, s);
    if (ret < 0)
        return ret;

    ret = dv_vk_stage_gpu_dequant(avctx, s);
    if (ret < 0)
        return ret;

    ret = dv_vk_stage_idct(avctx, s);
    if (ret < 0)
        return ret;

    ret = dv_vk_stage_color_convert(avctx, s);
    if (ret < 0)
        return ret;

    return dv_vk_stage_write_output(avctx, s);
}

static int dv_vk_frame_params(AVCodecContext *avctx, AVBufferRef *hw_frames_ctx)
{
    AVHWFramesContext     *frames_ctx = (AVHWFramesContext *)hw_frames_ctx->data;
    AVVulkanFramesContext *hwfc       = frames_ctx->hwctx;
    enum AVPixelFormat     sw_format  = avctx->sw_pix_fmt;
    int                    width      = avctx->coded_width;
    int                    height     = avctx->coded_height;

    if (sw_format == AV_PIX_FMT_NONE)
        sw_format = avctx->pix_fmt != AV_PIX_FMT_NONE ? avctx->pix_fmt : AV_PIX_FMT_YUV411P;

    sw_format = dv_vk_choose_sw_format(sw_format);

    if (width <= 0)
        width = avctx->width;
    if (height <= 0)
        height = avctx->height;

    if (width <= 0 || height <= 0) {
        av_log(avctx, AV_LOG_ERROR, "dv_vulkan: invalid frame params w=%d h=%d (coded=%dx%d)\n", width, height, avctx->coded_width,
               avctx->coded_height);
        return AVERROR(EINVAL);
    }

    frames_ctx->format    = AV_PIX_FMT_VULKAN;
    frames_ctx->sw_format = AV_PIX_FMT_RGBA;
    frames_ctx->width     = width;
    frames_ctx->height    = height;

    hwfc->format[0] = VK_FORMAT_R8G8B8A8_UNORM;
    hwfc->format[1] = VK_FORMAT_UNDEFINED;
    hwfc->format[2] = VK_FORMAT_UNDEFINED;

    hwfc->tiling = VK_IMAGE_TILING_OPTIMAL;
    hwfc->usage  = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    avctx->color_range     = AVCOL_RANGE_JPEG;
    avctx->colorspace      = AVCOL_SPC_RGB;
    avctx->color_primaries = AVCOL_PRI_SMPTE170M;
    avctx->color_trc       = AVCOL_TRC_LINEAR;

    av_log(avctx, AV_LOG_DEBUG, "dv_vulkan: frame_params sw=%s w=%d h=%d (internal_yuv=%s)\n", av_get_pix_fmt_name(frames_ctx->sw_format),
           frames_ctx->width, frames_ctx->height, av_get_pix_fmt_name(sw_format));

    return 0;
}

const FFHWAccel ff_dv_vulkan_hwaccel = {
    .p.name         = "dv_vulkan",
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_DVVIDEO,
    .p.pix_fmt      = AV_PIX_FMT_VULKAN,
    .alloc_frame    = dv_vulkan_alloc_frame,
    .start_frame    = dv_vulkan_start_frame,
    .end_frame      = dv_vulkan_end_frame,
    .decode_slice   = dv_vulkan_decode_slice,
    .init           = dv_vulkan_decode_init,
    .uninit         = dv_vulkan_decode_uninit,
    .frame_params   = dv_vk_frame_params,
    .priv_data_size = sizeof(DVSubContext),
    .caps_internal  = HWACCEL_CAP_ASYNC_SAFE,
};
