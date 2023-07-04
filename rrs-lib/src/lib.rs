// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! RISC-V instruction set simulator library
//!
//! Containts the building blocks for a RISC-V ISS. The seperate rrs-cli uses rrs-lib to implement
//! a CLI driven ISS.

pub mod csrs;
pub mod instruction_executor;
pub mod instruction_formats;
pub mod instruction_string_outputter;
pub mod memories;
pub mod process_instruction;

use downcast_rs::{impl_downcast, Downcast};

pub use process_instruction::process_instruction;

/// A trait for objects which do something with RISC-V instructions (e.g. execute them or print a
/// disassembly string).
///
/// There is one function per RISC-V instruction. Each function takes the appropriate struct from
/// [instruction_formats] giving access to the decoded fields of the instruction. All functions
/// return the [InstructionProcessor::InstructionResult] associated type.
pub trait InstructionProcessor {
    type InstructionResult;

    fn process_add(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sub(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sll(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_slt(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sltu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_xor(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_srl(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_sra(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_or(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_and(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_addi(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_slli(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_slti(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_sltui(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_xori(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_srli(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_srai(
        &mut self,
        dec_insn: instruction_formats::ITypeShamt,
    ) -> Self::InstructionResult;
    fn process_ori(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_andi(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_lui(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult;
    fn process_auipc(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult;

    fn process_beq(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bne(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_blt(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bltu(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bge(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;
    fn process_bgeu(&mut self, dec_insn: instruction_formats::BType) -> Self::InstructionResult;

    fn process_lb(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lbu(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lh(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lhu(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;
    fn process_lw(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_sb(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;
    fn process_sh(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;
    fn process_sw(&mut self, dec_insn: instruction_formats::SType) -> Self::InstructionResult;

    fn process_jal(&mut self, dec_insn: instruction_formats::JType) -> Self::InstructionResult;
    fn process_jalr(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_mul(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulh(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulhu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_mulhsu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_div(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_divu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_rem(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;
    fn process_remu(&mut self, dec_insn: instruction_formats::RType) -> Self::InstructionResult;

    fn process_fence(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult;

    fn process_csrrw(&mut self, dec_insn: instruction_formats::ITypeCSR)
        -> Self::InstructionResult;
    fn process_csrrs(&mut self, dec_insn: instruction_formats::ITypeCSR)
        -> Self::InstructionResult;
    fn process_csrrc(&mut self, dec_insn: instruction_formats::ITypeCSR)
        -> Self::InstructionResult;
    fn process_csrrwi(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;
    fn process_csrrsi(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;
    fn process_csrrci(
        &mut self,
        dec_insn: instruction_formats::ITypeCSR,
    ) -> Self::InstructionResult;

    fn process_ecall(&mut self) -> Self::InstructionResult;
    fn process_ebreak(&mut self) -> Self::InstructionResult;
    fn process_wfi(&mut self) -> Self::InstructionResult;
    fn process_mret(&mut self) -> Self::InstructionResult;
}

/// State of a single RISC-V hart (hardware thread)
pub struct HartState {
    /// x1 - x31 register values. The contents of index 0 (the x0 zero register) are ignored.
    pub registers: [u32; 32],
    /// Program counter
    pub pc: u32,
    /// Gives index of the last register written if one occurred in the previous instruciton. Set
    /// to `None` if latest instruction did not write a register.
    pub last_register_write: Option<usize>,

    pub priv_level: csrs::PrivLevel,
    pub csr_set: csrs::CSRSet,
}

impl HartState {
    pub fn new() -> Self {
        HartState {
            registers: [0; 32],
            pc: 0,
            last_register_write: None,
            priv_level: csrs::PrivLevel::M,
            csr_set: csrs::CSRSet::default(),
        }
    }

    /// Write a register in the hart state. Used by executing instructions for correct zero
    /// register handling
    fn write_register(&mut self, reg_index: usize, data: u32) {
        if reg_index == 0 {
            return;
        }

        self.registers[reg_index] = data;
        self.last_register_write = Some(reg_index)
    }

    /// Read a register from the hart state. Used by executing instructions for correct zero
    /// register handling
    fn read_register(&self, reg_index: usize) -> u32 {
        if reg_index == 0 {
            0
        } else {
            self.registers[reg_index]
        }
    }

    fn write_csr(&mut self, csr_addr: u32, data: u32) -> bool {
        if let Some(csr) = self.csr_set.get_csr(csr_addr) {
            csr.write(data);
            true
        } else {
            false
        }
    }

    // TODO: get_csr needs &mut so this needs &mut, can we refactor to avoid this without having to
    // make two copies of get_csr?
    fn read_csr(&mut self, csr_addr: u32) -> Option<u32> {
        let csr = self.csr_set.get_csr(csr_addr)?;

        Some(csr.read())
    }
}

impl Default for HartState {
    fn default() -> Self {
        Self::new()
    }
}

/// The different sizes used for memory accesses
#[derive(Clone, Copy)]
pub enum MemAccessSize {
    /// 8 bits
    Byte,
    /// 16 bits
    HalfWord,
    /// 32 bits
    Word,
}

/// A trait for objects which implement memory operations
pub trait Memory: Downcast {
    /// Read `size` bytes from `addr`.
    ///
    /// `addr` must be aligned to `size`.
    /// Returns `None` if `addr` doesn't exist in this memory.
    fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32>;

    /// Write `size` bytes of `store_data` to `addr`
    ///
    /// `addr` must be aligned to `size`.
    /// Returns `true` if write succeeds.
    fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool;
}

pub trait CSR {
    fn read(&self) -> u32;
    fn write(&mut self, val: u32);
}

impl_downcast!(Memory);

#[cfg(test)]
mod tests {
    use super::csrs::ExceptionCause;
    use super::instruction_executor::{InstructionExecutor, InstructionTrap};
    use super::instruction_string_outputter::InstructionStringOutputter;
    use super::*;

    fn run_insns<'a, M: Memory>(executor: &mut InstructionExecutor<'a, M>, end_pc: u32) {
        while executor.hart_state.pc != end_pc {
            let mut outputter = InstructionStringOutputter {
                insn_pc: executor.hart_state.pc,
            };
            let insn_bits = executor
                .mem
                .read_mem(executor.hart_state.pc, MemAccessSize::Word)
                .unwrap();

            assert_eq!(executor.step(), Ok(()));

            println!(
                "{:x} {}",
                executor.hart_state.pc,
                process_instruction(&mut outputter, insn_bits).unwrap()
            );
            if let Some(reg_index) = executor.hart_state.last_register_write {
                println!(
                    "x{} = {:08x}",
                    reg_index, executor.hart_state.registers[reg_index]
                );
            }
        }
    }

    #[test]
    fn test_insn_execute() {
        let mut hart_state = HartState::new();
        let mut mem = memories::VecMemory::new(vec![
            0x1234b137, 0xbcd10113, 0xf387e1b7, 0x3aa18193, 0xbed892b7, 0x7ac28293, 0x003100b3,
            0xf4e0e213, 0x02120a63, 0x00121463, 0x1542c093, 0x00c0036f, 0x0020f0b3, 0x402080b3,
            0x00000397, 0x02838393, 0x0003a403, 0x00638483, 0x0023d503, 0x00139223, 0x0043a583,
            0xdeadfaa1, 0x00000000, 0x00000000, 0xdeadbeef, 0xbaadf00d,
        ]);

        hart_state.pc = 0;

        let mut executor = InstructionExecutor {
            hart_state: &mut hart_state,
            mem: &mut mem,
        };

        run_insns(&mut executor, 0x54);

        assert_eq!(
            executor.step(),
            Err(InstructionTrap::Exception(
                ExceptionCause::IllegalInstruction,
                0xdeadfaa1
            ))
        );

        assert_eq!(hart_state.registers[1], 0x05bc8f77);
        assert_eq!(hart_state.registers[2], 0x1234abcd);
        assert_eq!(hart_state.registers[3], 0xf387e3aa);
        assert_eq!(hart_state.registers[4], 0xffffff7f);
        assert_eq!(hart_state.registers[5], 0xbed897ac);
        assert_eq!(hart_state.registers[6], 0x00000030);
        assert_eq!(hart_state.registers[7], 0x00000060);
        assert_eq!(hart_state.registers[8], 0xdeadbeef);
        assert_eq!(hart_state.registers[9], 0xffffffad);
        assert_eq!(hart_state.registers[10], 0x0000dead);
        assert_eq!(hart_state.registers[11], 0xbaad8f77);
    }

    #[test]
    fn test_csr_insn() {
        let mut hart = HartState::new();
        /*
         * li x1, 0xdeadbeef
         * csrw mscratch, x1
         * csrrc x2, mscratch, 0x1f
         * li x3, 0xbaadf00d
         * csrrw x4, mscratch, x3
         * li x5, 0x1234abc0
         * csrw mtval, x5
         * csrrs x6, mtval, 0x1
         */
        let mut mem = memories::VecMemory::new(vec![
            0xdeadc0b7, 0xeef08093, 0x34009073, 0x340ff173, 0xbaadf1b7, 0x00d18193, 0x34019273,
            0x1234b2b7, 0xbc028293, 0x30529073, 0x3050e373,
        ]);

        hart.pc = 0;

        let mut executor = InstructionExecutor {
            hart_state: &mut hart,
            mem: &mut mem,
        };

        run_insns(&mut executor, 0x2c);

        assert_eq!(hart.registers[1], 0xdeadbeef);
        assert_eq!(hart.registers[2], 0xdeadbeef);
        assert_eq!(hart.registers[3], 0xbaadf00d);
        assert_eq!(hart.registers[4], 0xdeadbee0);
        assert_eq!(hart.registers[5], 0x1234abc0);
        assert_eq!(hart.registers[6], 0x1234abc0);

        assert_eq!(hart.csr_set.mscratch.val, 0xbaadf00d);
        assert_eq!(hart.csr_set.mtvec.base, 0x1234abc0);
        assert_eq!(hart.csr_set.mtvec.vectored_mode, true);
    }
}
