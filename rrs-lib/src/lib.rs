mod instruction_executor;
mod instruction_formats;
mod instruction_string_outputter;
mod memories;
mod process_instruction;

use downcast_rs::{impl_downcast, Downcast};

pub use process_instruction::process_instruction;

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
}

pub struct HartState {
    pub registers: [u32; 32],
    pub pc: u32,
    pub last_register_write: Option<usize>,
}

impl HartState {
    fn new() -> Self {
        HartState {
            registers: [0; 32],
            pc: 0,
            last_register_write: None,
        }
    }

    fn write_register(&mut self, reg_index: usize, data: u32) {
        if reg_index == 0 {
            return;
        }

        self.registers[reg_index] = data;
        self.last_register_write = Some(reg_index)
    }

    fn read_register(&self, reg_index: usize) -> u32 {
        if reg_index == 0 {
            0
        } else {
            self.registers[reg_index]
        }
    }
}

#[derive(Clone, Copy)]
pub enum MemAccessSize {
    Byte,     // 8 bits
    HalfWord, // 16 bits
    Word,     // 32 bits
}

pub trait Memory: Downcast {
    fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32>;
    fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool;
}

impl_downcast!(Memory);

#[cfg(test)]
mod tests {
    use super::instruction_executor::{InstructionException, InstructionExecutor};
    use super::instruction_string_outputter::InstructionStringOutputter;
    use super::*;

    #[test]
    fn test_insn_execute() {
        let mut hart = HartState::new();
        let mut mem = memories::VecMemory::new(vec![
            0x1234b137, 0xbcd10113, 0xf387e1b7, 0x3aa18193, 0xbed892b7, 0x7ac28293, 0x003100b3,
            0xf4e0e213, 0x02120a63, 0x00121463, 0x1542c093, 0x00c0036f, 0x0020f0b3, 0x402080b3,
            0x00000397, 0x02838393, 0x0003a403, 0x00638483, 0x0023d503, 0x00139223, 0x0043a583,
            0x00000000, 0x00000000, 0x00000000, 0xdeadbeef, 0xbaadf00d,
        ]);

        hart.pc = 0;

        // TODO: With the 'executor' concept we need to effectively create a new one each step as
        // it's meant to be just taking a reference to things to execute, but then if we want to
        // access those things we either do it via the executor or create a new one before the next
        // step to allow access via the 'main' object, could just make step part of the 'main'
        // object? Having the executor only coupled to a bare minimum of state could be good?
        let mut executor = InstructionExecutor {
            hart_state: &mut hart,
            mem: &mut mem,
        };

        while executor.hart_state.pc != 0x54 {
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

        assert_eq!(executor.hart_state.registers[1], 0x05bc8f77);
        assert_eq!(executor.hart_state.registers[2], 0x1234abcd);
        assert_eq!(executor.hart_state.registers[3], 0xf387e3aa);
        assert_eq!(executor.hart_state.registers[4], 0xffffff7f);
        assert_eq!(executor.hart_state.registers[5], 0xbed897ac);
        assert_eq!(executor.hart_state.registers[6], 0x00000030);
        assert_eq!(executor.hart_state.registers[7], 0x00000060);
        assert_eq!(executor.hart_state.registers[8], 0xdeadbeef);
        assert_eq!(executor.hart_state.registers[9], 0xffffffad);
        assert_eq!(executor.hart_state.registers[10], 0x0000dead);
        assert_eq!(executor.hart_state.registers[11], 0xbaad8f77);

        assert_eq!(
            executor.step(),
            Err(InstructionException::IllegalInstruction(0x54, 0))
        );
    }
}
