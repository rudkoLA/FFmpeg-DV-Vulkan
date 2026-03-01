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


#include "config.h"
#include "config_components.h"
#include <errno.h>

#include "libavutil/avassert.h"
#include "libavutil/internal.h"
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

static int dv_vulkan_decode_init(AVCodecContext *avctx)
{
    AVHWDeviceContext *device_ctx;
    AVVulkanDeviceContext *vk_dev;

    av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: DV Vulkan init requested\n");

    if (!avctx->hw_device_ctx) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: Missing hw_device_ctx\n");
        return AVERROR(EINVAL);
    }

    device_ctx = (AVHWDeviceContext *)avctx->hw_device_ctx->data;
    if (!device_ctx || device_ctx->type != AV_HWDEVICE_TYPE_VULKAN) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: Invalid Vulkan device context\n");
        return AVERROR(EINVAL);
    }

    vk_dev = device_ctx->hwctx;
    if (!vk_dev) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: Missing AVVulkanDeviceContext\n");
        return AVERROR(EINVAL);
    }

    av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: Found Vulkan Device Context\n");
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: DV Vulkan hwaccel is not yet implemented and exits gracefully\n");

    return AVERROR(ENOSYS);
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
    .p.name      = "dv_vulkan",
    .p.type      = AVMEDIA_TYPE_VIDEO,
    .p.id        = AV_CODEC_ID_DVVIDEO,
    .p.pix_fmt   = AV_PIX_FMT_VULKAN,
    .start_frame   = dv_vulkan_start_frame,
    .end_frame     = dv_vulkan_end_frame,
    .decode_slice  = dv_vulkan_decode_slice,
    .init          = dv_vulkan_decode_init,
    .uninit        = ff_vk_decode_uninit,
    .frame_params  = ff_vk_frame_params,
    .priv_data_size = sizeof(FFVulkanDecodeContext),
    .caps_internal  = HWACCEL_CAP_THREAD_SAFE,
};
