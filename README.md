# BladeBit Chia Plotter

[![Release Builds](https://github.com/Chia-Network/bladebit/actions/workflows/build-release.yml/badge.svg?branch=master&event=push)](https://github.com/Chia-Network/bladebit/actions/workflows/build-release.yml)

A high-performance **k32-only**, Chia (XCH) plotter supporting in-RAM and disk-based plotting.

## Requirements

### In-RAM
**416 GiB of RAM are required** to run it, and a few more megabytes for stack space and small allocations.

64-bit is supported only, for obvious reasons.


### Disk-based
A minimum of **4 GiB of RAM** is required, with lower bucket counts requiring up to 12 GiB of RAM. 

Around **480 GiB** of total temporary space is required when plotting to disk in the default mode, or around 390 GiB with `--alternate` mode enabled.

The exact amounts of RAM and disk space required may vary slightly depending on the system's page size and the target disk file system block size (block-alignment is required for direct I/O).

SSDs are highly recommended for disk-based plotting.


## Prerequisites
Linux, Windows and MacOS (both intel and ARM (Apple Silicon)) are supported.


### Linux

Install the following packages:
```bash
# CentOS or RHEL-based
sudo yum group install -y "Development Tools"
sudo yum install -y cmake gmp-devel numactl-devel

# Ubuntu or Debian-based
sudo apt install -y build-essential cmake libgmp-dev libnuma-dev
```

### Windows
Must have at least [Visual Studio 2019](https://visualstudio.microsoft.com/vs/) or its build tools installed.

### macOS
Must have Xcode or Xcode build tools installed.
`brew install cmake`

Optionally install `gmp`:
`brew install gmp`


## Building

```bash

# Clone the repo & its submodules
git clone https://github.com/Chia-Network/bladebit.git && cd bladebit

# Create a build directory for cmake and cd into it
mkdir -p build && cd build

# Generate config files & build
cmake ..
cmake --build . --target bladebit --config Release
```

The resulting binary will be found under the `build/` directory.
On Windows it will be under `build/Release/`.

## Usage
Run **bladebit** with the `-h` for complete usage and command line options:

```bash
# Linux & macOS
build/bladebit -h

# Windows
build/Release/bladebit.exe -h
```


The bladebit CLI uses the format `bladebit <GLOBAL_OPTIONS> <sub_command> <COMMAND_OPTIONS>`.

Use the aforementioned `-h` parameter to get the full list of sub-commands and `GLOBAL_OPTIONS`. 
The `sub_command`-specific `COMMAND_OPTIONS` can be obtained by using the `help` sub command with the desired command as the parameter: 

```bash
bladebit help ramplot
bladebit help diskplot
```

### In-RAM
Basic `ramplot` usage:
```bash
# OG plots
./bladebit -f <farmer_public_key> -p <pool_public_key> ramplot <output_directory>

# Portable plots
./bladebit -f <farmer_public_key> -c <pool_contract_address> ramplot <output_directory>
```

### Disk-Based
Basic `diskplot` usage:
```bash

# OG plots
./bladebit -f <farmer_public_key> -p <pool_public_key> diskplot -t1 <temp_directory> <output_directory>

# Portable plots
./bladebit -f <farmer_public_key> -c <pool_contract_address> diskplot -t1 <temp_directory> <output_directory>

# Differing temp directories:
./bladebit -f ... -c ... diskplot -t1 /path/to/temp_1 -t2 /path/to/temp2 /my/output/dir

# With a 100 GiB temp2 cache and alternating mode
./bladebit -f ... -c ... diskplot -a --cache 100G -t1 /path/to/temp_1 -t2 /path/to/temp2 /my/output/dir

# With fine-grained thread control depending on the workload
./bladebit -f ... -c ... diskplot --f1-threads 12 --fp-threads 32 -t1 /path/to/temp_1  /my/output/dir
```


## License
Licensed under the [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). See [LICENSE](LICENSE).


Copyright 2023 Harold Brenes, Chia Network Inc

