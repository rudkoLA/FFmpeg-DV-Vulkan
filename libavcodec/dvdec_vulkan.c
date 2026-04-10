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

#include "libavutil/mem.h"
#include "libavutil/vulkan_spirv.h"

typedef struct DVSubContext {
    FFVulkanDecodeContext vk_dec;
    FFVulkanShader        shd;

    FFVulkanDescriptorSetBinding input_desc[1];
    FFVulkanDescriptorSetBinding output_desc[2];
} DVSubContext;


static int dv_vk_init_descriptor_sets(FFVulkanContext *s, FFVulkanShader *shd)
{
    int err;

    /* SET 0: INPUT — сирий DV бітстрим */
    RET(ff_vk_shader_add_descriptor_set(s, shd,
        (FFVulkanDescriptorSetBinding[]) {
            {
                .name        = "dv_input_bitstream",
                .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "uint8_t data[];",
            },
            {
                .name        = "dv_quant_tables",
                .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
                .buf_content = "int16_t quant[];",
            },
        },
        /* nb                  */ 2,
        /* singular            */ 0,
        /* print_to_shader_only*/ 0));

    /* SET 1: OUTPUT */
    RET(ff_vk_shader_add_descriptor_set(s, shd,
        (FFVulkanDescriptorSetBinding[]) {
            {
                .name    = "out_luma",
                .type    = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .stages  = VK_SHADER_STAGE_COMPUTE_BIT,
                .elems   = 1,
                .mem_layout = "r8",      /* ← додати формат */
                .mem_quali  = "writeonly",
                .dimensions = 2,
            },
            {
                .name    = "out_chroma",
                .type    = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .stages  = VK_SHADER_STAGE_COMPUTE_BIT,
                .elems   = 1,
                .mem_layout = "rg8",     /* ← додати формат */
                .mem_quali  = "writeonly",
                .dimensions = 2,
            },
        }, 2, 0, 0));

    return 0;
fail:
    return err;
}


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
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: hw_device_ctx=%p hw_frames_ctx=%p\n",
        avctx->hw_device_ctx, avctx->hw_frames_ctx);

    if (!avctx->hw_device_ctx && !avctx->hw_frames_ctx)
    {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: No hw_device_ctx and no hw_frames_ctx!\n");
        return AVERROR(EINVAL);
    }

    int err;

    // 1. Try to initialize. If it fails, log it but let's see why.
    // Замість ff_vk_decode_init використовуємо ff_vk_init напряму
    AVHWFramesContext *frames_ctx;
    avctx->hw_frames_ctx = av_hwframe_ctx_alloc(avctx->hw_device_ctx);
    if (!avctx->hw_frames_ctx)
        return AVERROR(ENOMEM);

    frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
    frames_ctx->format    = AV_PIX_FMT_VULKAN;
    frames_ctx->sw_format = avctx->sw_pix_fmt != AV_PIX_FMT_NONE
                            ? avctx->sw_pix_fmt : AV_PIX_FMT_YUV420P;
    frames_ctx->width     = avctx->coded_width;
    frames_ctx->height    = avctx->coded_height;
    frames_ctx->initial_pool_size = 4;

    av_log(avctx, AV_LOG_ERROR,
       "VULKAN_DEBUG: frames_ctx: sw_format=%s w=%d h=%d\n",
       av_get_pix_fmt_name(frames_ctx->sw_format),
       frames_ctx->width, frames_ctx->height);

    err = av_hwframe_ctx_init(avctx->hw_frames_ctx);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: av_hwframe_ctx_init failed (%d): %s\n",
            err, av_err2str(err));
        av_buffer_unref(&avctx->hw_frames_ctx);
        return err;
    }
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: hw_frames_ctx created OK\n");

    // Алокуємо shared_ctx вручну
    s->vk_dec.shared_ctx = av_mallocz(sizeof(*s->vk_dec.shared_ctx));
    if (!s->vk_dec.shared_ctx)
        return AVERROR(ENOMEM);

    err = ff_vk_init(&s->vk_dec.shared_ctx->s, avctx, NULL, avctx->hw_frames_ctx);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: ff_vk_init failed (%d)\n", err);
        av_freep(&s->vk_dec.shared_ctx);
        return err;
    }
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: ff_vk_init OK\n");

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
                            NULL, 0, 8, 8, 1, 0);

    if (err >= 0)
    {
        av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: Shader initialized successfully!\n");
    }

    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: calling descriptor_sets\n");
    err = dv_vk_init_descriptor_sets(vk_ctx, &s->shd);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: descriptor_sets failed (%d): %s\n",
            err, av_err2str(err));
        return err;
    }
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: descriptor_sets OK\n");

    // 3. GLSL Definition
    FFVulkanShader *shd = &s->shd;
    GLSLC(0, void main() {                                                      );
    GLSLC(0,     ivec2 pos = ivec2(gl_GlobalInvocationID.xy);                  );
    GLSLC(0,     imageStore(out_luma[0],   pos, vec4(0.0));                    );
    GLSLC(0,     imageStore(out_chroma[0], pos, vec4(0.5, 0.5, 0.0, 0.0));    );
    GLSLC(0, } );                                                                      
    
    // 4. Link
    uint8_t *spv_data;
    size_t   spv_len;
    void    *spv_opaque = NULL;

    FFVkSPIRVCompiler *spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: ff_vk_spirv_init failed\n");
        return AVERROR(ENOMEM);
    }
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: spirv_init OK\n");

    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: shader src:\n%s\n", shd->src.str);

    err = spv->compile_shader(vk_ctx, spv, &s->shd,
                            &spv_data, &spv_len, "main", &spv_opaque);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: compile_shader failed (%d): %s\n",
            err, av_err2str(err));
        spv->uninit(&spv);
        return err;
    }
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: compile_shader OK\n");

    err = ff_vk_shader_link(vk_ctx, &s->shd, spv_data, spv_len, "main");
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);
    spv->uninit(&spv);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: shader_link failed (%d): %s\n",
            err, av_err2str(err));
        return err;
    }
    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: shader_link OK — init complete!\n");
    return 0;
}

static int dv_vulkan_start_frame(AVCodecContext *avctx,
                                 const AVBufferRef *hw_frames_ctx,
                                 const uint8_t *buffer, uint32_t size)
{
    av_log(avctx, AV_LOG_INFO, "VULKAN_DEBUG: Start frame\n");
    return 0;
}


static int dv_vk_bind_and_dispatch(AVCodecContext *avctx,
                                   AVFrame        *output_frame,
                                   FFVkBuffer     *input_buf,
                                   int width, int height)
{
    DVSubContext      *s   = avctx->internal->hwaccel_priv_data;
    FFVulkanContext   *vk  = &s->vk_dec.shared_ctx->s;
    int err;

    /* bind SET 0: input buffer */
    RET(ff_vk_shader_update_desc_buffer(vk, NULL, &s->shd,
                                        0, 0, 0,
                                        input_buf, VK_WHOLE_SIZE, 0,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));

    /* bind SET 1 binding 0: luma */
    ff_vk_shader_update_img_array(vk, NULL, &s->shd,
                                output_frame, NULL,
                                1, 0,
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_NULL_HANDLE);

    /* bind SET 1 binding 1: chroma */
    ff_vk_shader_update_img_array(vk, NULL, &s->shd,
                                output_frame, NULL,
                                1, 1,
                                VK_IMAGE_LAYOUT_GENERAL,
                                VK_NULL_HANDLE);

    /* dispatch */
    ff_vk_exec_bind_shader(vk, NULL, &s->shd);
    vk->vkfn.CmdDispatch(/* cmd */ NULL,
                         (width  + 7) / 8,
                         (height + 7) / 8, 1);
    return 0;
fail:
    return err;
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

static int dv_vk_frame_params(AVCodecContext *avctx, AVBufferRef *hw_frames_ctx)
{
    AVHWFramesContext *frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;
    AVVulkanFramesContext *hwfc   = frames_ctx->hwctx;

    frames_ctx->format     = AV_PIX_FMT_VULKAN;
    frames_ctx->sw_format  = avctx->sw_pix_fmt;
    frames_ctx->width      = avctx->coded_width;
    frames_ctx->height     = avctx->coded_height;

    /* NV12: plane 0 = Y (R8), plane 1 = UV (R8G8) */
    hwfc->format[0] = VK_FORMAT_R8_UNORM;
    hwfc->format[1] = VK_FORMAT_R8G8_UNORM;

    av_log(avctx, AV_LOG_ERROR, "VULKAN_DEBUG: dv_vk_frame_params OK\n");
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
    .frame_params = dv_vk_frame_params,
    .priv_data_size = sizeof(DVSubContext),
    .caps_internal = HWACCEL_CAP_ASYNC_SAFE | HWACCEL_CAP_THREAD_SAFE,
};
