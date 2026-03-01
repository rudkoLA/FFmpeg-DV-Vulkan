# FFmpeg DV Video Vulkan Hardware Acceleration

Base FFmpeg README can be found [here](FFMPEG_README.md).

## Installation
GNU Make >= 3.81 is needed to compile FFmpeg.

Most importantly you need to install a Vulkan version >= 1.3.277. It is recommended to do so using https://vulkan.lunarg.com/sdk/home as other means may install older non supported version (e.g. on Ubuntu 24.04, `sudo apt install libvulkan1` will install Vulkan version 1.3.275 which is incompatible with FFmpeg).

Other dependencies may include `libvulkan-dev vulkan-tools vulkan-headers glslang-tools libglslang-dev` or their equivalents on other operating systems. But these will most likely be explained by FFmpeg, while the vulkan one may be difficult to track down.

## Compiling

To fully compile FFmpeg with Vulkan hardware acceleration use the following commands:

```
./configure --enable-vulkan
make
```

Afterwards to not recompile everything every time you make a change, you can use the following command to only recompile based on the changed files:

```
make -j$(nproc)
```

After either of these you can use the following to install the FFmpeg binary globally on your system:
```
make install
```

## Usage

To encode any video to the DV format you can use the following command:

```
ffmpeg -i [file name] -target ntsc-dv input.dv
```

Then to decode using **software** decoding you may use the following:
```
ffmpeg -i input.dv -c:v mpeg4 -qscale:v 3 -c:a aac output.mp4
```

And currently, as proper Vulkan decoding is not yet implemented, you should use the following in the meantime to test decoding DV using Vulkan:

```
./ffmpeg -init_hw_device vulkan -hwaccel vulkan -i input.dv -f null -
```

Additionally `-v debug` flag may be helpful to further debug console output.