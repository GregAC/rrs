/* Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
   Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
   SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

OUTPUT_ARCH(riscv)

_mem_size = 0x100000;
_mem_start = 0x100000;
_mem_end = _mem_start + _mem_size;

_stack_start = _mem_end;

SECTIONS
{
  . = _mem_start;
  .text.start : { *(.text.start) }
  .text : { *(.text) }
  .data : { *(.data) *(.sdata) }
  .= ALIGN(4);
  _bss_start = .;
  .bss : { *(.bss) *(.sbss) }
  .= ALIGN(4);
  _bss_end = .;
}
