# Getting Started with Interdevcopy

# Introduction
Interdevcopy: inter-device data copy library  
Interdevcopy is a library for data transfer between devices including host and various accelerators.

## Requirements
The following environment is required to use Inter-Device Copy.

- Hardware: x86 server with vector engine
   - Vector Engine: VE10, VE10E and VE20
   - GPU: NVIDIA
- Operating System: RedHat Enterprise Linux / CentOS / Rocky Linux 8.4 or above
- VE driver, VEOS and AVEO with VE-GPU direct transfer support
 (preview release version) released on GitHub
   - https://github.com/veos-sxarr-NEC/veos/releases
- CUDA 11.7.0 or higher

## Build
### Install from RPM
You can install Inter-Device Copy by running the following commands.

- Install from your local computer
    1. Download [the rpm packages](https://github.com/SX-Aurora/interdevcopy/releases) from GitHub.  
At this time, download two of interdevcopy, interdevcopy-devel.
    2. Put the rpm packages to your any directory.
    3. Install the local rpm packages via yum command.

```
$ sudo yum install <path_to_interdevcopy_rpm> <path_to_interdevcopy-devel_rpm>
```
### Install from source (with building)
Packages required for building
- To build interdevcopy, AVEO with VE-GPU direct transfer support
 (veoffload-aveo, veoffload-aveo-devel and veoffload-aveorun-ve1) and CUDA 11 are required.
- It's using GNU make for library interdevcopy build and GNU Autoconf/Automake/Libtool for generating Makefile etc.
- It has been confirmed that the build can be done in the following conditions.
   - Compiler GCC: gcc-c++-8.5.0
   - GNU make: make-4.2.1
   - GNU Autoconf: autoconf-2.69
   - GNU Automake: automake-1.16.1
   - Libtool: libtool-2.4.6

Specify the configure option as follows and execute make install.

```
$ git clone https://github.com/SX-Aurora/interdevcopy.git
$ cd interdevcopy
$ ./bootstrap
$ ./configure --prefix=/opt/nec/interdevcopy --libdir=/opt/nec/interdevcopy/lib64 \
    --with-veo=/opt/nec/ve/veos --with-cuda=/usr/local/cuda
$ make
$ sudo make install
```
