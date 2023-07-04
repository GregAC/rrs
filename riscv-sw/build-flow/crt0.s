# Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
# Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.section .text.start

_start:
  .global start

  la x1, _bss_start
  la x2, _bss_end
  beq x1, x2, main_entry

zero_bss_loop:
  sw x0, 0(x1)
  add x1, x1, 4
  bne x1, x2, zero_bss_loop

main_entry:
  la sp, _stack_start
  jal ra, main

  la x1, 0x80000004
  li x2, 1
  sw x2, 0(x1)
