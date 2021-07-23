// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

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
//! use rrs_lib::instruction_executor::{InstructionExecutor, InstructionException};
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
//! assert_eq!(executor.step(), Err(InstructionException::FetchError(0xc)));
//! ```

use super::instruction_formats;
use super::process_instruction;
use super::{HartState, InstructionProcessor, MemAccessSize, Memory};
use paste::paste;

/// Different exceptions that can occur during instruction execution
#[derive(Debug, PartialEq)]
pub enum InstructionException {
    // TODO: Better to name the fields?
    IllegalInstruction(u32, u32),
    FetchError(u32),
    LoadAccessFault(u32),
    StoreAccessFault(u32),
    AlignmentFault(u32),
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
        self.hart_state.write_register(dec_insn.rd, result)
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
    ) -> Result<(), InstructionException> {
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
            return Err(InstructionException::AlignmentFault(addr));
        }

        // Attempt to read data from memory, returning a LoadAccessFault as an error if it is not.
        let mut load_data = match self.mem.read_mem(addr, size) {
            Some(d) => d,
            None => {
                return Err(InstructionException::LoadAccessFault(addr));
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
    ) -> Result<(), InstructionException> {
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
            return Err(InstructionException::AlignmentFault(addr));
        }

        // Write store data to memory, returning a StoreAccessFault as an error if write fails.
        if self.mem.write_mem(addr, size, data) {
            Ok(())
        } else {
            Err(InstructionException::StoreAccessFault(addr))
        }
    }

    /// Execute instruction pointed to by `hart_state.pc`
    ///
    /// Returns `Ok` where instruction execution was successful. `Err` with the relevant
    /// [InstructionException] is returned when the instruction execution causes an exception.
    pub fn step(&mut self) -> Result<(), InstructionException> {
        self.hart_state.last_register_write = None;

        // Fetch next instruction from memory
        if let Some(next_insn) = self.mem.read_mem(self.hart_state.pc, MemAccessSize::Word) {
            // Execute the instruction
            let step_result = process_instruction(self, next_insn);

            match step_result {
                Some(Ok(pc_updated)) => {
                    if !pc_updated {
                        // Instruction didn't update PC so increment to next instruction
                        self.hart_state.pc += 4;
                    }
                    Ok(())
                }
                // Instruction produced an error so return it
                Some(Err(e)) => Err(e),
                // Instruction decode failed so return an IllegalInstruction as an error
                None => Err(InstructionException::IllegalInstruction(
                    self.hart_state.pc,
                    next_insn,
                )),
            }
        } else {
            // Return a FetchError as an error if instruction fetch fails
            Err(InstructionException::FetchError(self.hart_state.pc))
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

impl<'a, M: Memory> InstructionProcessor for InstructionExecutor<'a, M> {
    /// Result is `Ok` when instruction execution is successful. `Ok(true) indicates the
    /// instruction updated the PC and Ok(false) indicates it did not (so the PC must be
    /// incremented to execute the next instruction).
    type InstructionResult = Result<bool, InstructionException>;

    make_alu_op_fns! {add, |a, b| a.wrapping_add(b)}
    make_alu_op_reg_fn! {sub, |a, b| a.wrapping_sub(b)}
    make_alu_op_fns! {slt, |a, b| if (a as i32) < (b as i32) {1} else {0}}
    make_alu_op_fns! {sltu, |a, b| if a < b {1} else {0}}
    make_alu_op_fns! {or, |a, b| a | b}
    make_alu_op_fns! {and, |a, b| a & b}
    make_alu_op_fns! {xor, |a, b| a ^ b}

    make_shift_op_fns! {sll, |a, b| a << b}
    make_shift_op_fns! {srl, |a, b| a >> b}
    make_shift_op_fns! {sra, |a, b| ((a as i32) >> b) as u32}

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

    fn process_beq(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a == b))
    }

    fn process_bne(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a != b))
    }

    fn process_blt(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| (a as i32) < (b as i32)))
    }

    fn process_bltu(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a < b))
    }

    fn process_bge(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| (a as i32) >= (b as i32)))
    }

    fn process_bgeu(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a >= b))
    }

    fn process_lb(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::Byte, true)?;

        Ok(false)
    }

    fn process_lbu(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::Byte, false)?;

        Ok(false)
    }

    fn process_lh(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::HalfWord, true)?;

        Ok(false)
    }

    fn process_lhu(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::HalfWord, false)?;

        Ok(false)
    }

    fn process_lw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::Word, false)?;

        Ok(false)
    }

    fn process_sb(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult {
        self.execute_store(dec_insn, MemAccessSize::Byte)?;

        Ok(false)
    }

    fn process_sh(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult {
        self.execute_store(dec_insn, MemAccessSize::HalfWord)?;

        Ok(false)
    }

    fn process_sw(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult {
        self.execute_store(dec_insn, MemAccessSize::Word)?;

        Ok(false)
    }

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
}
