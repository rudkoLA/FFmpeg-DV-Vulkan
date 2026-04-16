# FFmpeg DV Video Vulkan Hardware Acceleration

Base FFmpeg README can be found [here](FFMPEG_README.md).

## Installation
GNU Make >= 3.81 is needed to compile FFmpeg.

Most importantly you need a Vulkan install that is new enough for FFmpeg's Vulkan code. On Ubuntu 24.04, `sudo apt install libvulkan1` installs Vulkan 1.3.275, which is older than what ffmpeg expects.

The easiest way to work around that is to install the LunarG Vulkan SDK (1.3 prefered over 1.4) from the `.tar.xz` package and use it only for the current shell or build session. The SDK can coexist with the system Vulkan packages.

Typical dependencies are `libvulkan-dev vulkan-tools vulkan-headers glslang-tools libshaderc-dev` or their equivalents on your platform. If FFmpeg's configure step reports that the SPIR-V compiler backend is missing, install either `libshaderc` or `libglslang` support as well.

### Vulkan SDK workaround

If you already have a LunarG SDK tarball, unpack it and point the build to it:

```bash
cd ~/Downloads
mkdir -p ~/VulkanSDK
tar -xJf vulkansdk-linux-x86_64-<VERSION>.tar.xz -C ~/VulkanSDK
cd ~/VulkanSDK/<VERSION>  # adjust if your SDK layout differs
source x86_64/setup-env.sh  # if present
```

If your tarball does not ship `setup-env.sh`, set the paths manually instead:

```bash
export VULKAN_SDK="$HOME/VulkanSDK/<VERSION>/x86_64"
export PATH="$VULKAN_SDK/bin:$PATH"
export LD_LIBRARY_PATH="$VULKAN_SDK/lib:${LD_LIBRARY_PATH:-}"
export CPPFLAGS="-I$VULKAN_SDK/include ${CPPFLAGS:-}"
export LDFLAGS="-L$VULKAN_SDK/lib ${LDFLAGS:-}"
```

To verify the SDK is in use, check the header version and toolchain paths:

```bash
grep -E "VK_HEADER_VERSION|VK_HEADER_VERSION_COMPLETE" "$VULKAN_SDK/include/vulkan/vulkan_core.h"
pkg-config --modversion vulkan
vulkaninfo --summary | grep -E "Vulkan Instance Version|apiVersion"
```

Note that `pkg-config --modversion vulkan` can still report the system Vulkan package if the SDK does not ship a `vulkan.pc` file. In that case the headers and libraries from `VULKAN_SDK` are still usable through the exported include and library paths above.

## Compiling

To build FFmpeg with Vulkan hardware acceleration, use the SDK environment from above in the same shell and then run:

```bash
./configure \
  --enable-vulkan \
  --enable-libshaderc \
  --enable-x86asm \
  --enable-gpl \
  --enable-nonfree \
  --extra-cflags="-I$VULKAN_SDK/include" \
  --extra-ldflags="-L$VULKAN_SDK/lib" \
  --extra-libs="-lshaderc_combined -lstdc++ -lpthread -lm -ldl"
```

If `libshaderc` is not available on your system, `--enable-libglslang` is the alternative. The key point is that one SPIR-V compiler backend must be enabled so FFmpeg can build Vulkan shaders without falling back to the broken external path.

Once configure, run the following to fully compile ffmpeg:
```bash
make
```

Afterwards to not recompile everything every time you make a change, you can use the following command to only recompile based on the changed files:

```bash
make -j$(nproc)
```

After either of these you can use the following to install the FFmpeg binary globally on your system:
```bash
make install
```

## Usage

To encode any video to the DV format you can use the following command:

```bash
./ffmpeg -i <FILENAME> -target ntsc-dv input.dv
```

Then to decode using **software** decoding you may use the following:
```bash
./ffmpeg -i input.dv -c:v mpeg4 -qscale:v 3 -c:a aac output.mp4
```

To test DV decoding through the Vulkan path, use:

```bash
./ffmpeg -init_hw_device vulkan -hwaccel vulkan -i input.dv -c:v mpeg4 -qscale:v 3 -c:a aac output.mp4
```

`-v debug` is helpful if you want to see shader compilation, Vulkan backend selection, and CPU fallback messages.
