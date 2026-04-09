/*
 * DV decoder
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
 * Many thanks to Dan Dennedy <dan@dennedy.org> for providing wealth
 * of DV technical info.
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
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * DV decoder
 */

#define FF_INTERNAL_FIELDS
#define INDENT_shd INDENT(0)
#include "config.h"
#include "config_components.h"
#include <errno.h>

#include "libavutil/avassert.h"
#include "libavutil/internal.h"
#include "libavcodec/avcodec.h"
#include "libavutil/mem_internal.h"
#include "libavutil/thread.h"

#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"
#include "dv.h"
#include "dv_internal.h"
#include "dv_profile_internal.h"
#include "dvdata.h"
#include "get_bits.h"
#include "idctdsp.h"
#include "put_bits.h"
#include "simple_idct.h"
#include "thread.h"

#include "hwaccels.h"
#include "vulkan_decode.h"
#include "libavutil/vulkan_functions.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_vulkan.h"
#include "libavutil/vulkan.h"

typedef struct DVSubContext
{
    FFVulkanDecodeContext vk_dec;
    FFVulkanShader shd;
    struct FFVulkanPipeline *pl;
} DVSubContext;

static int dv_vulkan_decode_init(AVCodecContext *avctx)
{
    // 1. Try to get private data directly from avctx
    DVSubContext *s = avctx->internal->hwaccel_priv_data;

    // 2. If that's null, check the internal pointer
    if (!s && avctx->internal)
        s = avctx->internal->hwaccel_priv_data;

    // 3. If it's STILL null, we have to stop before we crash
    if (!s)
    {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: No hwaccel_priv_data found!\n");
        return AVERROR(EINVAL);
    }

    // Now proceed with your checks
    if (!avctx->hw_device_ctx)
    {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: No hw_device_ctx found!\n");
        return AVERROR(EINVAL);
    }

    int err;

    // 1. Try to initialize. If it fails, log it but let's see why.
    err = ff_vk_decode_init(avctx);
    if (err < 0)
    {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: ff_vk_decode_init failed (%d)\n", err);
        return err;
    }

    // 2. The Pointer Path Fix
    // If shared_ctx is null, we might need to access the context from the device_ctx instead
    FFVulkanContext *vk_ctx;
    if (s->vk_dec.shared_ctx)
    {
        vk_ctx = &s->vk_dec.shared_ctx->s;
    }
    else
    {
        // Fallback: try to get it from the hardware device context
        AVHWDeviceContext *device_ctx = (AVHWDeviceContext *)avctx->hw_device_ctx->data;
        AVVulkanDeviceContext *vk_dev = device_ctx->hwctx;
        // In some versions, you can use the device context to init the shader
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: shared_ctx is NULL, shader init might fail\n");
        return AVERROR(EINVAL);
    }

    // 3. Shader Init
    err = ff_vk_shader_init(&s->vk_dec.shared_ctx->s, &s->shd, "dv_compute",
                            VK_SHADER_STAGE_COMPUTE_BIT,
                            NULL, 0, "main", 8, 8, 1);

    if (err >= 0)
    {
        av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: Shader initialized successfully!\n");
    }

    // 3. GLSL Definition
    FFVulkanShader *shd = &s->shd;
    GLSLC(0, layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;);
    GLSLC(0, void main() {);
    GLSLC(0,     uint idx = gl_GlobalInvocationID.x;);
    GLSLC(0, });

    // 4. Link
    return ff_vk_shader_link(vk_ctx, &s->shd, NULL, 0, NULL);
}

static int dv_vulkan_start_frame(AVCodecContext *avctx,
                                 const AVBufferRef *hw_frames_ctx,
                                 const uint8_t *buffer, uint32_t size)
{
    av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: Start frame\n");
    return 0;
}

static int dv_vulkan_end_frame(AVCodecContext *avctx)
{
    av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: Ending frame\n");
    return 0;
}

static int dv_vulkan_decode_slice(AVCodecContext *avctx, const uint8_t *data, uint32_t size)
{
    av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: Decoding frame\n");
    return 0;
}

const FFHWAccel ff_dv_vulkan_hwaccel = {
    .p.name = "dv_vulkan",
    .p.type = AVMEDIA_TYPE_VIDEO,
    .p.id = AV_CODEC_ID_DVVIDEO,
    .p.pix_fmt = AV_PIX_FMT_VULKAN,
    .start_frame = dv_vulkan_start_frame,
    .end_frame = dv_vulkan_end_frame,
    .decode_slice = dv_vulkan_decode_slice,
    .init = dv_vulkan_decode_init,
    .uninit = ff_vk_decode_uninit,
    .frame_params = ff_vk_frame_params,
    .priv_data_size = sizeof(DVSubContext),
    .caps_internal = HWACCEL_CAP_THREAD_SAFE,
    .priv_data_size = 0,
};
