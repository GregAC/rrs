// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


#include <stdint.h>

#define DEV_WRITE(addr, val) (*((volatile uint32_t *)(addr)) = val)
#define DEV_READ(addr, val) (*((volatile uint32_t *)(addr)))
#define OUTCHAR_DEV 0x80000000

void putchar(char c) {
  DEV_WRITE(OUTCHAR_DEV, (uint32_t)c);
}

void write_str(char* s) {
  while (*s) {
    putchar(*s++);
  }
}

int main(void) {
  write_str("Hello World RRS!\n");
}
