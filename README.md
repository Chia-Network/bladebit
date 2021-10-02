# BladeBit Chia Plotter

A fast **RAM-only**, **k32-only**, Chia plotter.

## Requirements
**416 GiB of RAM are required** to run it, plus a few more megabytes for stack space and small allocations. 

64-bit is supported only, for obvious reasons.

## Prerequisites
Only **Linux** & **Windows** are supported.

### Linux

Install the following packages:
```bash
# CentOS or Amazon Linux
sudo yum group install -y "Development Tools"
sudo yum install -y cmake gmp-devel numactl-devel

# Ubuntu or Debian-based
sudo apt install -y build-essential cmake libgmp-dev libnuma-dev
```

### Windows
Must have [Visual Studio 2019](https://visualstudio.microsoft.com/vs/) or its build tools installed.

## Building


```bash

# Clone the repo & its submodules
git clone https://github.com/Chia-Network/bladebit.git && cd bladebit

# Create a build directory for cmake and cd into it
mkdir build
cd build

# Generate config files & build
cmake ..
cmake --build . --target bladebit --config Release
```

The resulting binary will be found under the `build/` directory.
On Windows it will be under `build/Release/`.

## Usage
Run **bladebit** with the `-h` for complete usage and command line options:

```bash
# Linux
build/bladebit -h

# Windows
build/Release/bladebit.exe -h
```


## License
Licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). See [LICENSE](LICENSE).


# Other Details

## Disk I/O
Writes to disk only occur to the final plot file, and it is done sequentially, un-buffered, with direct I/O. This means that writes will be block-aligned. If you've gotten faster writes elsewhere in your drive than you will get with this, it is likely that it is using buffered writes, therefore it "finishes" before it actually finishes writing to disk. The kernel will handle the I/O in the background from cache (you can confirm this with tools such as iotop). The final writes here ought to pretty much saturate your sequential writes. Writes begin happening in the background at Phase 3 and will continue to do so, depending on the disk I/O throughput, through the next plot, if it did not finish beforehand. At some point in Phase 1 of the next plot, it might stall if it still has not finished writing to disk and a buffer it requires is still being written to disk. On the system I tested, there was no interruption when using an NVMe drive.


## Pool Plots
Pool plots are fully supported and tested against the chia-blockchain implementation. The community has also verified that pool plots are working properly and winning proofs with them.

## NUMA systems
Memory is bound on interleaved mode for NUMA systems which currently gives the best performance on systems with several nodes. This is the default behavior on NUMA systems, it can be disabled with with the `-m or --no-numa` switch.


## Huge TLBs
This is not supported yet. Some folks have reported some gains when using huge page sizes. Although this was something I wanted to test, I focused first instead on things that did not necessarily depended on system config. But I'd like to add support for it in the future (trivial from the development point of view, I have just not configured the test system with huge page sizes).

## Other Observations
This implementation is highly memory-bound so optimizing your system towards fast memory access is essential. CPUs with large caches will benefit as well.


Copyright 2021 Harold Brenes, Chia Network Inc

