use super::instruction_formats;
use super::InstructionProcessor;
use paste::paste;

pub struct InstructionStringOutputter {
    pub insn_pc: u32,
}

// TODO: Implement display for instruction formats then make generic string out macro that takes an
// instruction format type, have generic function for actual output? Macro just a convenience for
// avoiding too much boiler plate?
macro_rules! string_out_for_alu_reg_op {
    ($name:ident) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_formats::RType
            ) -> Self::InstructionResult {
                format!("{} x{}, x{}, x{}", stringify!($name), dec_insn.rd, dec_insn.rs1,
                    dec_insn.rs2)
            }
        }
    };
}

macro_rules! string_out_for_alu_imm_op {
    ($name:ident) => {
        paste! {
            fn [<process_ $name i>](
                &mut self,
                dec_insn: instruction_formats::IType
            ) -> Self::InstructionResult {
                format!("{}i x{}, x{}, {}", stringify!($name), dec_insn.rd, dec_insn.rs1,
                    dec_insn.imm)
            }
        }
    };
}

macro_rules! string_out_for_alu_imm_shamt_op {
    ($name:ident) => {
        paste! {
            fn [<process_ $name i>](
                &mut self,
                dec_insn: instruction_formats::ITypeShamt
            ) -> Self::InstructionResult {
                format!("{}i x{}, x{}, {}", stringify!($name), dec_insn.rd, dec_insn.rs1,
                    dec_insn.shamt)
            }
        }
    };
}

macro_rules! string_out_for_alu_ops {
    ($($name:ident),*) => {
        $(
            string_out_for_alu_reg_op! {$name}
            string_out_for_alu_imm_op! {$name}
        )*
    }
}

macro_rules! string_out_for_shift_ops {
    ($($name:ident),*) => {
        $(
            string_out_for_alu_reg_op! {$name}
            string_out_for_alu_imm_shamt_op! {$name}
        )*
    }
}

macro_rules! string_out_for_branch_ops {
    ($($name:ident),*) => {
        $(
            paste! {
                fn [<process_ $name>](
                    &mut self,
                    dec_insn: instruction_formats::BType
                ) -> Self::InstructionResult {
                    let branch_pc = self.insn_pc.wrapping_add(dec_insn.imm as u32);

                    format!("{} x{}, x{}, 0x{:08x}", stringify!($name), dec_insn.rs1, dec_insn.rs2,
                        branch_pc)
                }
            }
        )*
    }
}

macro_rules! string_out_for_load_ops {
    ($($name:ident),*) => {
        $(
            paste! {
                fn [<process_ $name>](
                    &mut self,
                    dec_insn: instruction_formats::IType
                ) -> Self::InstructionResult {
                    format!("{} x{}, {}(x{})", stringify!($name), dec_insn.rd, dec_insn.imm,
                        dec_insn.rs1)
                }
            }
        )*
    }
}

macro_rules! string_out_for_store_ops {
    ($($name:ident),*) => {
        $(
            paste! {
                fn [<process_ $name>](
                    &mut self,
                    dec_insn: instruction_formats::SType
                ) -> Self::InstructionResult {
                    format!("{} x{}, {}(x{})", stringify!($name), dec_insn.rs2, dec_insn.imm,
                        dec_insn.rs1)
                }
            }
        )*
    }
}

impl InstructionProcessor for InstructionStringOutputter {
    type InstructionResult = String;

    // TODO: Make one macro that takes all names as arguments and generates all the functions
    // together
    string_out_for_alu_ops! {add, slt, xor, or, and}
    string_out_for_alu_reg_op! {sltu}
    string_out_for_alu_reg_op! {sub}
    string_out_for_shift_ops! {sll, srl, sra}

    fn process_sltui(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        format!(
            "sltiu x{}, x{}, {}",
            dec_insn.rd, dec_insn.rs1, dec_insn.imm
        )
    }

    fn process_lui(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult {
        format!("lui x{}, 0x{:08x}", dec_insn.rd, dec_insn.imm)
    }

    fn process_auipc(&mut self, dec_insn: instruction_formats::UType) -> Self::InstructionResult {
        let final_imm = self.insn_pc.wrapping_add(dec_insn.imm as u32);
        format!("auipc x{}, 0x{:08x}", dec_insn.rd, final_imm)
    }

    string_out_for_branch_ops! {beq, bne, bge, bgeu, blt, bltu}
    string_out_for_load_ops! {lb, lbu, lh, lhu, lw}
    string_out_for_store_ops! {sb, sh, sw}

    fn process_jal(&mut self, dec_insn: instruction_formats::JType) -> Self::InstructionResult {
        let target_pc = self.insn_pc.wrapping_add(dec_insn.imm as u32);
        format!("jal x{}, 0x{:08x}", dec_insn.rd, target_pc)
    }

    fn process_jalr(&mut self, dec_insn: instruction_formats::IType) -> Self::InstructionResult {
        format!(
            "jalr x{}, 0x{:03x}(x{})",
            dec_insn.rd, dec_insn.imm, dec_insn.rs1
        )
    }

    string_out_for_alu_reg_op! {mul}
    string_out_for_alu_reg_op! {mulh}
    string_out_for_alu_reg_op! {mulhu}
    string_out_for_alu_reg_op! {mulhsu}

    string_out_for_alu_reg_op! {div}
    string_out_for_alu_reg_op! {divu}
    string_out_for_alu_reg_op! {rem}
    string_out_for_alu_reg_op! {remu}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::process_instruction;

    #[test]
    fn test_insn_string_output() {
        let mut outputter = InstructionStringOutputter { insn_pc: 0 };

        let test_insns = vec![
            0x07b60893, 0x24dba193, 0x06f63813, 0x14044f13, 0x7804e893, 0x1ea6fa13, 0x00511693,
            0x00f45713, 0x417dd213, 0x01798733, 0x40e18ab3, 0x009e1533, 0x00c02fb3, 0x014ab933,
            0x0175cd33, 0x014350b3, 0x41a753b3, 0x00566fb3, 0x01de7db3, 0xdeadb637, 0x00064897,
            0x04c004ef, 0x100183e7, 0x04d38263, 0x05349063, 0x03774e63, 0x03dbdc63, 0x035e6a63,
            0x0398f863, 0x04c18983, 0x07841b83, 0x1883a403, 0x03af4b03, 0x15acd883, 0x0d320923,
            0x18061323, 0x0b382523, 0x034684b3, 0x03679f33, 0x0324bbb3, 0x03d9a233, 0x03f549b3,
            0x02ee5133, 0x02a6e9b3, 0x02c976b3,
        ];

        assert_eq!(
            process_instruction(&mut outputter, test_insns[0]),
            Some(String::from("addi x17, x12, 123"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[1]),
            Some(String::from("slti x3, x23, 589"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[2]),
            Some(String::from("sltiu x16, x12, 111"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[3]),
            Some(String::from("xori x30, x8, 320"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[4]),
            Some(String::from("ori x17, x9, 1920"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[5]),
            Some(String::from("andi x20, x13, 490"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[6]),
            Some(String::from("slli x13, x2, 5"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[7]),
            Some(String::from("srli x14, x8, 15"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[8]),
            Some(String::from("srai x4, x27, 23"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[9]),
            Some(String::from("add x14, x19, x23"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[10]),
            Some(String::from("sub x21, x3, x14"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[11]),
            Some(String::from("sll x10, x28, x9"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[12]),
            Some(String::from("slt x31, x0, x12"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[13]),
            Some(String::from("sltu x18, x21, x20"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[14]),
            Some(String::from("xor x26, x11, x23"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[15]),
            Some(String::from("srl x1, x6, x20"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[16]),
            Some(String::from("sra x7, x14, x26"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[17]),
            Some(String::from("or x31, x12, x5"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[18]),
            Some(String::from("and x27, x28, x29"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[19]),
            Some(String::from("lui x12, 0xdeadb000"))
        );

        outputter.insn_pc = 0x50;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[20]),
            Some(String::from("auipc x17, 0x00064050"))
        );

        outputter.insn_pc = 0x54;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[21]),
            Some(String::from("jal x9, 0x000000a0"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[22]),
            Some(String::from("jalr x7, 0x100(x3)"))
        );

        outputter.insn_pc = 0x5c;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[23]),
            Some(String::from("beq x7, x13, 0x000000a0"))
        );

        outputter.insn_pc = 0x60;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[24]),
            Some(String::from("bne x9, x19, 0x000000a0"))
        );

        outputter.insn_pc = 0x64;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[25]),
            Some(String::from("blt x14, x23, 0x000000a0"))
        );

        outputter.insn_pc = 0x68;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[26]),
            Some(String::from("bge x23, x29, 0x000000a0"))
        );

        outputter.insn_pc = 0x6c;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[27]),
            Some(String::from("bltu x28, x21, 0x000000a0"))
        );

        outputter.insn_pc = 0x70;
        assert_eq!(
            process_instruction(&mut outputter, test_insns[28]),
            Some(String::from("bgeu x17, x25, 0x000000a0"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[29]),
            Some(String::from("lb x19, 76(x3)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[30]),
            Some(String::from("lh x23, 120(x8)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[31]),
            Some(String::from("lw x8, 392(x7)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[32]),
            Some(String::from("lbu x22, 58(x30)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[33]),
            Some(String::from("lhu x17, 346(x25)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[34]),
            Some(String::from("sb x19, 210(x4)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[35]),
            Some(String::from("sh x0, 390(x12)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[36]),
            Some(String::from("sw x19, 170(x16)"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[37]),
            Some(String::from("mul x9, x13, x20"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[38]),
            Some(String::from("mulh x30, x15, x22"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[39]),
            Some(String::from("mulhu x23, x9, x18"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[40]),
            Some(String::from("mulhsu x4, x19, x29"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[41]),
            Some(String::from("div x19, x10, x31"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[42]),
            Some(String::from("divu x2, x28, x14"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[43]),
            Some(String::from("rem x19, x13, x10"))
        );

        assert_eq!(
            process_instruction(&mut outputter, test_insns[44]),
            Some(String::from("remu x13, x18, x12"))
        );
    }
}
