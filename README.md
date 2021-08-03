# Rust RISC-V Simulator (rrs)

A RISC-V ISS implemented in Rust. Supports the RV32IM instruction set (no
exceptions or CSRs yet).

RRS is split into two parts, rrs-lib and rrs-cli

## Quickstart

To build an example hello word program and run it on rrs-cli (requires a riscv
toolchain):

```sh
> make -C ./riscv-sw/examples/hello_world all
> cd rrs-cli
> cargo run -- --binary ../riscv-sw/examples/hello_world/hello_world.bin -l run.log
```

After execution `run.log` will contain a trace of ISS execution.

## rrs-lib

This crate implements the components required for the ISS. Documentation can be
generated via rustdoc

```sh
> cd rrs-lib
> cargo doc
```

Once generated it can be found in `target/doc/rrs-lib`

### Tests

`rrs-lib` implements a small number of tests, these can be run with cargo as
usual.

```sh
> cd rrs-lib
> cargo test
```

## rrs-cli

This crate implements a CLI for the ISS, build and run rrs-cli with a `--help`
argument to get full usage instructions.

```sh
> cd rrs-cli
> cargo run -- --help
```

E.g. to execute a binary `prog.bin` in a 1 MB memory starting at 0x20000000 

```sh
> cargo run -- --load_addr 20000000 --mem_size 100000 --binary prog.bin
```

rrs-cli has two special memory locations for output and controlling the
simulation.

* 0x80000000 - Write a character here to output it to stdout
* 0x80000004 - Write a non zero value here to terminate the simulation

## RISC-V software

A RISC-V makefile based build flow with  can be found in `riscv-sw/build-flow`.
A RISC-V toolchain will be required, such as the ones available from lowRISC:
https://github.com/lowRISC/lowrisc-toolchains/releases.

The makefile assumes GCC with the prefix riscv32-unknown-elf (which the lowRISC
RV32IMC toolchain gives you). Adjust the relevant variables in
`riscv-sw/Makefile.include` if you have something different.

To build the hello_world example in the repository root run:

```sh
> make -C ./riscv-sw/examples/hello_world all
```

This will build a `.elf`, `.bin` and `.dis` file. The `.bin` is suitable for
running on rrs-cli and the `.dis` is a disassembly.

### Using the build flow

To build a new program create a subdirectory under `sw/` and create a `Makfile`
with the following contents:

```
PROG_NAME = [program name here]

include ../Makefile.include # Adjust as required depending on sub-directory
```

Set `PROG_NAME` to whatever you want to call the binary. By default all .c and
.s files in the directory  will be built and linked together. The startup code
is in `riscv-sw/build-flow/crt0.s` and calls the function `main`. See
`riscv-sw/Makefile.include` for other variables that can be set to alter the
build.

## RISC-V Architectural Tests

rrs passes the M and I suites in the RISC-V architectural tests. To run the
suites:

```sh
> cd rrs-cli
> cargo build
> cd ..
> make -C ./vendor/riscv-arch-test/ TARGETDIR=`pwd`/riscv-sw/tests/riscv-arch-test-target RISCV_TARGET=rrs RISCV_DEVICE=M
> make -C ./vendor/riscv-arch-test/ TARGETDIR=`pwd`/riscv-sw/tests/riscv-arch-test-target RISCV_TARGET=rrs RISCV_DEVICE=I
```

## Blog

I've written about rrs on my blog, see:

* [Building a RISC-V Simulator in Rust - Part 1](https://gregchadwick.co.uk/blog/building-rrs-pt1/)
