// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! An [InstructionProcessor] that executes instructions.
//!
//! The [InstructionExecutor] takes a [HartState] and a [Memory]. The [HartState] is updated by the
//! instruction execution using the [Memory] for all memory accesses. A [InstructionExecutor::step]
//! function is provided to deal with reading the next instruction from the memory, updating the PC
//! appropriately and wraps the call to [process_instruction()]`.
//!
//! # Example
//!
//! ```
//! use rrs_lib::HartState;
//! use rrs_lib::memories::VecMemory;
//! use rrs_lib::instruction_executor::{InstructionExecutor, InstructionTrap};
//! use rrs_lib::csrs::ExceptionCause;
//!
//! let mut hart = HartState::new();
//! // Memory contains these instructions:
//! // lui x2, 0x1234b
//! // lui x3, 0xf387e
//! // add x1, x2, x3
//! let mut mem = VecMemory::new(vec![0x1234b137, 0xf387e1b7, 0x003100b3]);
//!
//! hart.pc = 0;
//!
//! let mut executor = InstructionExecutor {
//!     hart_state: &mut hart,
//!     mem: &mut mem,
//! };
//!
//! assert_eq!(executor.step(), Ok(()));
//! assert_eq!(executor.hart_state.registers[2], 0x1234b000);
//! assert_eq!(executor.step(), Ok(()));
//! assert_eq!(executor.hart_state.registers[3], 0xf387e000);
//! assert_eq!(executor.step(), Ok(()));
//! assert_eq!(executor.hart_state.registers[1], 0x05bc9000);
//! // Memory only contains three instructions so next step will produce a fetch error
//! assert_eq!(
//!     executor.step(),
//!     Err(InstructionTrap::Exception(
//!         ExceptionCause::InstructionAccessFault,
//!         0x0
//!     ))
//! );
//! ```

use super::csrs::{CSRAddr, ExceptionCause, MIx, PrivLevel};
use super::instruction_formats;
use super::process_instruction;
use super::CSR;
use super::{HartState, InstructionProcessor, MemAccessSize, Memory};
use paste::paste;

/// Different traps that can occur during instruction execution
#[derive(Debug, PartialEq)]
pub enum InstructionTrap {
    /// Trap is a synchronous exception, with a particular cause, u32 is used as mtval.
    Exception(ExceptionCause, u32),
    /// Trap is an asynchronous interrupt with a particular number.
    Interrupt(u32),
}

/// An `InstructionProcessor` that execute instructions, updating `hart_state` as appropriate.
pub struct InstructionExecutor<'a, M: Memory> {
    /// Memory used by load and store instructions
    pub mem: &'a mut M,
    pub hart_state: &'a mut HartState,
}

impl<'a, M: Memory> InstructionExecutor<'a, M> {
    fn execute_reg_reg_op<F>(&mut self, dec_insn: instruction_formats::RType, op: F)
    where
        F: Fn(u32, u32) -> u32,
    {
        let a = self.hart_state.read_register(dec_insn.rs1);
        let b = self.hart_state.read_register(dec_insn.rs2);
        let result = op(a, b);
        self.hart_state.write_register(dec_insn.rd, result);
    }

    fn execute_reg_imm_op<F>(&mut self, dec_insn: instruction_formats::IType, op: F)
    where
        F: Fn(u32, u32) -> u32,
    {
        let a = self.hart_state.read_register(dec_insn.rs1);
        let b = dec_insn.imm as u32;
        let result = op(a, b);
        self.hart_state.write_register(dec_insn.rd, result);
    }

    fn execute_reg_imm_shamt_op<F>(&mut self, dec_insn: instruction_formats::ITypeShamt, op: F)
    where
        F: Fn(u32, u32) -> u32,
    {
        let a = self.hart_state.read_register(dec_insn.rs1);
        let result = op(a, dec_insn.shamt);
        self.hart_state.write_register(dec_insn.rd, result);
    }

    fn execute_csr_op<F>(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
        use_imm: bool,
        op: F,
    ) -> Result<(), InstructionTrap>
    where
        F: Fn(u32, u32) -> u32,
    {
        let old_csr = self
            .hart_state
            .read_csr(dec_insn.csr)
            .ok_or(InstructionTrap::Exception(
                ExceptionCause::IllegalInstruction,
                0,
            ))?;

        let a = if use_imm {
            dec_insn.rs1 as u32
        } else {
            self.hart_state.read_register(dec_insn.rs1)
        };

        let new_csr = op(old_csr, a);

        if !self.hart_state.write_csr(dec_insn.csr, new_csr) {
            panic!("CSR write should succeed if execution reaches this point");
        }

        self.hart_state.write_register(dec_insn.rd, old_csr);

        Ok(())
    }

    // Returns true if branch succeeds
    fn execute_branch<F>(&mut self, dec_insn: instruction_formats::BType, cond: F) -> bool
    where
        F: Fn(u32, u32) -> bool,
    {
        let a = self.hart_state.read_register(dec_insn.rs1);
        let b = self.hart_state.read_register(dec_insn.rs2);

        if cond(a, b) {
            let new_pc = self.hart_state.pc.wrapping_add(dec_insn.imm as u32);
            self.hart_state.pc = new_pc;
            true
        } else {
            false
        }
    }

    fn execute_load(
        &mut self,
        dec_insn: instruction_formats::IType,
        size: MemAccessSize,
        signed: bool,
    ) -> Result<(), InstructionTrap> {
        let addr = self
            .hart_state
            .read_register(dec_insn.rs1)
            .wrapping_add(dec_insn.imm as u32);

        // Determine if address is aligned to size, returning an AlignmentFault as an error if it
        // is not.
        let align_mask = match size {
            MemAccessSize::Byte => 0x0,
            MemAccessSize::HalfWord => 0x1,
            MemAccessSize::Word => 0x3,
        };

        if (addr & align_mask) != 0x0 {
            return Err(InstructionTrap::Exception(
                ExceptionCause::LoadAddressMisaligned,
                addr,
            ));
        }

        // Attempt to read data from memory, returning a LoadAccessFault as an error if it is not.
        let mut load_data = match self.mem.read_mem(addr, size) {
            Some(d) => d,
            None => {
                return Err(InstructionTrap::Exception(
                    ExceptionCause::LoadAccessFault,
                    addr,
                ));
            }
        };

        // Sign extend loaded data if required
        if signed {
            load_data = (match size {
                MemAccessSize::Byte => (load_data as i8) as i32,
                MemAccessSize::HalfWord => (load_data as i16) as i32,
                MemAccessSize::Word => load_data as i32,
            }) as u32;
        }

        // Write load data to destination register
        self.hart_state.write_register(dec_insn.rd, load_data);
        Ok(())
    }

    fn execute_store(
        &mut self,
        dec_insn: instruction_formats::SType,
        size: MemAccessSize,
    ) -> Result<(), InstructionTrap> {
        let addr = self
            .hart_state
            .read_register(dec_insn.rs1)
            .wrapping_add(dec_insn.imm as u32);
        let data = self.hart_state.read_register(dec_insn.rs2);

        let align_mask = match size {
            MemAccessSize::Byte => 0x0,
            MemAccessSize::HalfWord => 0x1,
            MemAccessSize::Word => 0x3,
        };

        // Determine if address is aligned to size, returning an AlignmentFault as an error if it
        // is not.
        if (addr & align_mask) != 0x0 {
            return Err(InstructionTrap::Exception(
                ExceptionCause::StoreAddressMisaligned,
                addr,
            ));
        }

        // Write store data to memory, returning a StoreAccessFault as an error if write fails.
        if self.mem.write_mem(addr, size, data) {
            Ok(())
        } else {
            Err(InstructionTrap::Exception(
                ExceptionCause::StoreAccessFault,
                addr,
            ))
        }
    }

    pub fn pending_interrupt(&self) -> Option<u32> {
        if self.hart_state.csr_set.mstatus.mie {
            let pending_interrupt_bits =
                self.hart_state.csr_set.mip.read() & self.hart_state.csr_set.mie.read();

            if pending_interrupt_bits != 0 {
                let mut pending_interrupts: MIx = Default::default();
                pending_interrupts.write(pending_interrupt_bits);

                // TODO: Add constants/enum for interrupt numbers?
                if pending_interrupts.external {
                    Some(11)
                } else if pending_interrupts.software {
                    Some(3)
                } else if pending_interrupts.timer {
                    Some(7)
                } else {
                    panic!("Unknown interrupt bit set {:08x}", pending_interrupt_bits)
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Execute instruction pointed to by `hart_state.pc`
    ///
    /// Returns `Ok` where instruction execution was successful. `Err` with the relevant
    /// [InstructionTrap] is returned when the instruction execution causes a trap.
    pub fn step(&mut self) -> Result<(), InstructionTrap> {
        self.hart_state.last_register_write = None;

        if let Some(irq) = self.pending_interrupt() {
            Err(InstructionTrap::Interrupt(irq))
        } else if let Some(next_insn) = self.mem.read_mem(self.hart_state.pc, MemAccessSize::Word) {
            // Fetch next instruction from memory and eecute the instruction if fetch was
            // successful
            let step_result = process_instruction(self, next_insn);

            match step_result {
                Some(Ok(pc_updated)) => {
                    if !pc_updated {
                        // Instruction didn't update PC so increment to next instruction
                        self.hart_state.pc += 4;
                    }
                    Ok(())
                }
                // Instruction produced an illegal instruction error or decode failed so return an
                // IllegalInstruction as an error, supplying instruction bits
                Some(Err(InstructionTrap::Exception(ExceptionCause::IllegalInstruction, _)))
                | None => Err(InstructionTrap::Exception(
                    ExceptionCause::IllegalInstruction,
                    next_insn,
                )),
                // Instruction produced an error so return it
                Some(Err(e)) => Err(e),
            }
        } else {
            // Return a FetchError as an error if instruction fetch fails
            // TODO: Give PC as mtval?
            Err(InstructionTrap::Exception(
                ExceptionCause::InstructionAccessFault,
                0,
            ))
        }
    }

    pub fn handle_trap(&mut self, trap: InstructionTrap) {
        self.hart_state.csr_set.mepc.val = self.hart_state.pc;
        self.hart_state.csr_set.mstatus.mpie = self.hart_state.csr_set.mstatus.mie;
        self.hart_state.csr_set.mstatus.mie = false;
        self.hart_state.csr_set.mstatus.mpp = self.hart_state.priv_level;

        match trap {
            InstructionTrap::Interrupt(interrupt_num) => {
                self.hart_state.csr_set.mcause.cause = interrupt_num | 0x80000000;
                self.hart_state.csr_set.mtval.val = 0;

                if self.hart_state.csr_set.mtvec.vectored_mode {
                    self.hart_state.pc = self.hart_state.csr_set.mtvec.base + interrupt_num * 4;
                } else {
                    self.hart_state.pc = self.hart_state.csr_set.mtvec.base;
                }
            }
            InstructionTrap::Exception(cause, val) => {
                self.hart_state.csr_set.mcause.cause = cause.into();
                self.hart_state.csr_set.mtval.val = val;
                self.hart_state.pc = self.hart_state.csr_set.mtvec.base;
            }
        }
    }
}

fn sign_extend_u32(x: u32) -> i64 {
    (x as i32) as i64
}

// Macros to implement various repeated operations (e.g. ALU reg op reg instructions).
macro_rules! make_alu_op_reg_fn {
    ($name:ident, $op_fn:expr) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_formats::RType
            ) -> Self::InstructionResult {
                self.execute_reg_reg_op(dec_insn, $op_fn);

                Ok(false)
            }
        }
    };
}

macro_rules! make_alu_op_imm_fn {
    ($name:ident, $op_fn:expr) => {
        paste! {
            fn [<process_ $name i>](
                &mut self,
                dec_insn: instruction_formats::IType
            ) -> Self::InstructionResult {
                self.execute_reg_imm_op(dec_insn, $op_fn);

                Ok(false)
            }
        }
    };
}

macro_rules! make_alu_op_imm_shamt_fn {
    ($name:ident, $op_fn:expr) => {
        paste! {
            fn [<process_ $name i>](
                &mut self,
                dec_insn: instruction_formats::ITypeShamt
            ) -> Self::InstructionResult {
                self.execute_reg_imm_shamt_op(dec_insn, $op_fn);

                Ok(false)
            }
        }
    };
}

macro_rules! make_alu_op_fns {
    ($name:ident, $op_fn:expr) => {
        make_alu_op_reg_fn! {$name, $op_fn}
        make_alu_op_imm_fn! {$name, $op_fn}
    };
}

macro_rules! make_shift_op_fns {
    ($name:ident, $op_fn:expr) => {
        make_alu_op_reg_fn! {$name, $op_fn}
        make_alu_op_imm_shamt_fn! {$name, $op_fn}
    };
}

macro_rules! make_branch_op_fn {
    ($name:ident, $cond_fn:expr) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_formats::BType
            ) -> Self::InstructionResult {
                Ok(self.execute_branch(dec_insn, $cond_fn))
            }
        }
    };
}

macro_rules! make_load_op_fn_inner {
    ($name:ident, $size:ty, $signed: expr) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_formats::IType
            ) -> Self::InstructionResult {
                self.execute_load(dec_insn, $size, $signed)?;

                Ok(false)
            }
        }
    };
}

macro_rules! make_load_op_fn {
    ($name:ident, $size:ty, signed) => {
        make_load_op_fn_inner! {$name, $size, true}
    };
    ($name:ident, $size:ty, unsigned) => {
        make_load_op_fn_inner! {$name, $size, false}
    };
}

macro_rules! make_store_op_fn {
    ($name:ident, $size:ty) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_formats::SType
            ) -> Self::InstructionResult {
                self.execute_store(dec_insn, $size)?;

                Ok(false)
            }
        }
    };
}

macro_rules! make_csr_op_fns {
    ($name:ident, $op_fn:expr) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_formats::ITypeCSR
            ) -> Self::InstructionResult {
                self.execute_csr_op(dec_insn, false, $op_fn)?;

                Ok(false)
            }
        }

        paste! {
            fn [<process_ $name i>](
                &mut self,
                dec_insn: instruction_formats::ITypeCSR
            ) -> Self::InstructionResult {
                self.execute_csr_op(dec_insn, true, $op_fn)?;

                Ok(false)
            }
        }
    };
}

impl<'a, M: Memory> InstructionProcessor for InstructionExecutor<'a, M> {
    /// Result is `Ok` when instruction execution is successful. `Ok(true) indicates the
    /// instruction updated the PC and Ok(false) indicates it did not (so the PC must be
    /// incremented to execute the next instruction).
    type InstructionResult = Result<bool, InstructionTrap>;

    make_alu_op_fns! {add, |a, b| a.wrapping_add(b)}
    make_alu_op_reg_fn! {sub, |a, b| a.wrapping_sub(b)}
    make_alu_op_fns! {slt, |a, b| if (a as i32) < (b as i32) {1} else {0}}
    make_alu_op_fns! {sltu, |a, b| if a < b {1} else {0}}
    make_alu_op_fns! {or, |a, b| a | b}
    make_alu_op_fns! {and, |a, b| a & b}
    make_alu_op_fns! {xor, |a, b| a ^ b}

    make_shift_op_fns! {sll, |a, b| a << (b & 0x1f)}
    make_shift_op_fns! {srl, |a, b| a >> (b & 0x1f)}
    make_shift_op_fns! {sra, |a, b| ((a as i32) >> (b & 0x1f)) as u32}

    fn process_lui(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult {
        self.hart_state
            .write_register(dec_insn.rd, dec_insn.imm as u32);

        Ok(false)
    }

    fn process_auipc(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult {
        let result = self.hart_state.pc.wrapping_add(dec_insn.imm as u32);
        self.hart_state.write_register(dec_insn.rd, result);

        Ok(false)
    }

    make_branch_op_fn! {beq, |a, b| a == b}
    make_branch_op_fn! {bne, |a, b| a != b}
    make_branch_op_fn! {blt, |a, b|  (a as i32) < (b as i32)}
    make_branch_op_fn! {bltu, |a, b| a < b}
    make_branch_op_fn! {bge, |a, b|  (a as i32) >= (b as i32)}
    make_branch_op_fn! {bgeu, |a, b| a >= b}

    make_load_op_fn! {lb, MemAccessSize::Byte, signed}
    make_load_op_fn! {lbu, MemAccessSize::Byte, unsigned}
    make_load_op_fn! {lh, MemAccessSize::HalfWord, signed}
    make_load_op_fn! {lhu, MemAccessSize::HalfWord, unsigned}
    make_load_op_fn! {lw, MemAccessSize::Word, unsigned}

    make_store_op_fn! {sb, MemAccessSize::Byte}
    make_store_op_fn! {sh, MemAccessSize::HalfWord}
    make_store_op_fn! {sw, MemAccessSize::Word}

    fn process_jal(&mut self, dec_insn: instruction_formats::JType) -> Self::InstructionResult {
        let target_pc = self.hart_state.pc.wrapping_add(dec_insn.imm as u32);
        self.hart_state
            .write_register(dec_insn.rd, self.hart_state.pc + 4);
        self.hart_state.pc = target_pc;

        Ok(true)
    }

    fn process_jalr(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        let mut target_pc = self
            .hart_state
            .read_register(dec_insn.rs1)
            .wrapping_add(dec_insn.imm as u32);
        target_pc &= 0xfffffffe;

        self.hart_state
            .write_register(dec_insn.rd, self.hart_state.pc + 4);
        self.hart_state.pc = target_pc;

        Ok(true)
    }

    make_alu_op_reg_fn! {mul, |a, b| a.wrapping_mul(b)}
    make_alu_op_reg_fn! {mulh, |a, b| (sign_extend_u32(a).wrapping_mul(sign_extend_u32(b)) >> 32) as u32}
    make_alu_op_reg_fn! {mulhu, |a, b| (((a as u64).wrapping_mul(b as u64)) >> 32) as u32}
    make_alu_op_reg_fn! {mulhsu, |a, b| (sign_extend_u32(a).wrapping_mul(b as i64) >> 32) as u32}

    make_alu_op_reg_fn! {div, |a, b| if b == 0 {u32::MAX} else {((a as i32).wrapping_div(b as i32)) as u32}}
    make_alu_op_reg_fn! {divu, |a, b| if b == 0 {u32::MAX} else {a / b}}
    make_alu_op_reg_fn! {rem, |a, b| if b == 0 {a} else {((a as i32).wrapping_rem(b as i32)) as u32}}
    make_alu_op_reg_fn! {remu, |a, b| if b == 0 {a} else {a % b}}

    fn process_fence(&mut self, _dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        Ok(false)
    }

    make_csr_op_fns! {csrrw, |_old_csr, a| a}
    make_csr_op_fns! {csrrs, |old_csr, a| old_csr | a}
    make_csr_op_fns! {csrrc, |old_csr, a| old_csr & !a}

    fn process_mret(&mut self) -> Self::InstructionResult {
        self.hart_state.pc = self.hart_state.csr_set.mepc.val;
        self.hart_state.priv_level = self.hart_state.csr_set.mstatus.mpp;

        match self.hart_state.csr_set.mstatus.mpp {
            PrivLevel::M => {
                self.hart_state.csr_set.mstatus.mie = self.hart_state.csr_set.mstatus.mpie;
                self.hart_state.csr_set.mstatus.mpie = true;
                self.hart_state.csr_set.mstatus.mpp = PrivLevel::M;
            }
            _ => {
                panic!("mstatus.mpp should only be M mode");
            }
        }

        Ok(true)
    }

    fn process_wfi(&mut self) -> Self::InstructionResult {
        Ok(false)
    }

    fn process_ecall(&mut self) -> Self::InstructionResult {
        Err(InstructionTrap::Exception(ExceptionCause::ECallMMode, 0))
    }

    fn process_ebreak(&mut self) -> Self::InstructionResult {
        Err(InstructionTrap::Exception(ExceptionCause::Breakpoint, 0))
    }
}
