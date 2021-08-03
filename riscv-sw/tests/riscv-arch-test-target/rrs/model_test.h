#ifndef _COMPLIANCE_MODEL_H
#define _COMPLIANCE_MODEL_H

#define RRS_SIM_CHAR_DEV 0x80000000
#define RRS_SIM_CTRL_DEV 0x80000004

//TODO: Add code here to run after all tests have been run
// The .align 4 ensures that the signature begins at a 16-byte boundary
#define RVMODEL_HALT                                              \
  la x1, begin_signature; \
  la x6, RRS_SIM_CHAR_DEV; \
  li x5, 10; \
  sig_out: \
  lw x2, 0(x1); \
  li x3, 8; \
  sig_char_out: \
  srl x4, x2, 28; \
  bge x4, x5, sig_char_alpha; \
  add x4, x4, 48; \
  j sig_char_done; \
  sig_char_alpha: \
  add x4, x4, 87; \
  sig_char_done: \
  sw x4, 0(x6); \
  sll x2, x2, 4; \
  add x3, x3, -1; \
  bne x3, x0, sig_char_out; \
  li x2, 10; \
  sw x2, 0(x6); \
  add x1, x1, 4; \
  la x2, end_signature; \
  bne x1, x2, sig_out; \
\
  la x1, RRS_SIM_CTRL_DEV; \
  li x2, 1; \
  sw x2, 0(x1); \
  self_loop:  j self_loop;

//TODO: declare the start of your signature region here. Nothing else to be used here.
// The .align 4 ensures that the signature ends at a 16-byte boundary
#define RVMODEL_DATA_BEGIN                                              \
  .align 4; .global begin_signature; begin_signature:

//TODO: declare the end of the signature region here. Add other target specific contents here.
#define RVMODEL_DATA_END                                                      \
  .align 4; .global end_signature; end_signature:


//RVMODEL_BOOT
//TODO:Any specific target init code should be put here or the macro can be left empty

// For code that has a split rom/ram area
// Code below will copy from the rom area to ram the
// data.strings and .data sections to ram.
// Use linksplit.ld
#define RVMODEL_BOOT

#define RVMODEL_IO_WRITE_STR(_SP, _STR)

#define RVMODEL_IO_ASSERT_GPR_EQ(_SP, _R, _I)

//RVTEST_IO_ASSERT_SFPR_EQ
#define RVMODEL_IO_ASSERT_SFPR_EQ(_F, _R, _I)
//RVTEST_IO_ASSERT_DFPR_EQ
#define RVMODEL_IO_ASSERT_DFPR_EQ(_D, _R, _I)

// TODO: specify the routine for setting machine software interrupt
#define RVMODEL_SET_MSW_INT

// TODO: specify the routine for clearing machine software interrupt
#define RVMODEL_CLEAR_MSW_INT

// TODO: specify the routine for clearing machine timer interrupt
#define RVMODEL_CLEAR_MTIMER_INT

// TODO: specify the routine for clearing machine external interrupt
#define RVMODEL_CLEAR_MEXT_INT

#endif // _COMPLIANCE_MODEL_H

