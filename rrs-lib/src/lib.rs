mod instruction_format {
    pub const OPCODE_LOAD: u32 = 0x03;
    pub const OPCODE_MISC_MEM: u32 = 0x0f;
    pub const OPCODE_OP_IMM: u32 = 0x13;
    pub const OPCODE_AUIPC: u32 = 0x17;
    pub const OPCODE_STORE: u32 = 0x23;
    pub const OPCODE_OP: u32 = 0x33;
    pub const OPCODE_LUI: u32 = 0x37;
    pub const OPCODE_BRANCH: u32 = 0x63;
    pub const OPCODE_JALR: u32 = 0x67;
    pub const OPCODE_JAL: u32 = 0x6f;
    pub const OPCODE_SYSTEM: u32 = 0x73;

    #[derive(Debug, PartialEq)]
    pub struct RType {
        pub funct7: u32,
        pub rs2: usize,
        pub rs1: usize,
        pub funct3: u32,
        pub rd: usize
    }

    impl RType {
        pub fn new(insn: u32) -> RType {
            RType {
                funct7: (insn >> 25) & 0x7f,
                rs2: ((insn >> 20) & 0x1f) as usize,
                rs1: ((insn >> 15) & 0x1f) as usize,
                funct3: (insn >> 12) & 0x7,
                rd: ((insn >> 7) & 0x1f) as usize
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct IType {
        pub imm: i32,
        pub rs1: u32,
        pub funct3: u32,
        pub rd: u32
    }

    impl IType {
        pub fn new(insn: u32) -> IType {
            let uimm : i32 = ((insn >> 20) & 0x7ff) as i32;

            let imm : i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 11)
            } else {
                uimm
            };

            IType {
                imm: imm,
                rs1: (insn >> 15) & 0x1f,
                funct3: (insn >> 12) & 0x7,
                rd: (insn >> 7) & 0x1f
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct SType {
        pub imm: i32,
        pub rs2: u32,
        pub rs1: u32,
        pub funct3: u32,
    }

    impl SType {
        pub fn new(insn: u32) -> SType {
            let uimm : i32 = (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1f)) as i32;

            let imm : i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 11)
            } else {
                uimm
            };

            SType {
                imm: imm,
                rs2: (insn >> 20) & 0x1f,
                rs1: (insn >> 15) & 0x1f,
                funct3: (insn >> 12) & 0x7
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct BType {
        pub imm: i32,
        pub rs2: u32,
        pub rs1: u32,
        pub funct3: u32,
    }

    impl BType {
        pub fn new(insn: u32) -> BType {
            let uimm : i32 = (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1e) |
                              ((insn & 0x80) << 4)) as i32;

            let imm : i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 12)
            } else {
                uimm
            };

            println!("imm: {}", imm);

            BType {
                imm: imm,
                rs2: (insn >> 20) & 0x1f,
                rs1: (insn >> 15) & 0x1f,
                funct3: (insn >> 12) & 0x7
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct UType {
        pub imm: i32,
        pub rd: u32,
    }

    impl UType {
        pub fn new(insn: u32) -> UType {
            UType {
                imm: (insn & 0xffff_f000) as i32,
                rd: (insn >> 7) & 0x1f
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct JType {
        pub imm: i32,
        pub rd: u32,
    }

    impl JType {
        pub fn new(insn: u32) -> JType {
            let uimm : i32 = ((insn & 0xff000) | ((insn & 0x100000) >> 9) |
                              ((insn >> 20) & 0x7fe)) as i32;

            let imm : i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 20)
            } else {
                uimm
            };

            JType {
                imm: imm,
                rd: (insn >> 7) & 0x1f
            }
        }
    }
}

pub trait InstructionProcessor {
    type InstructionResult;

    fn process_add(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_sub(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
}

pub struct InstructionStringOutputter {}

impl InstructionProcessor for InstructionStringOutputter {
    type InstructionResult = String;

    fn process_add(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult {
        format!("add x{}, x{}, x{}", dec_insn.rd, dec_insn.rs1, dec_insn.rs2)
    }

    fn process_sub(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult {
        format!("sub x{}, x{}, x{}", dec_insn.rd, dec_insn.rs1, dec_insn.rs2)
    }
}

fn process_opcode_op<T: InstructionProcessor>(processor: &mut T, insn_bits: u32) ->
    Option<T::InstructionResult> {

    let dec_insn = instruction_format::RType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_add(dec_insn)),
                0b010_0000 => Some(processor.process_sub(dec_insn)),
                _ => None
            },
        /*0b001 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_sll(dec_insn)),
                _ => None
            },
        0b010 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_slt(dec_insn)),
                _ => None
            },
        0b011 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_sltu(dec_insn)),
                _ => None
            },
        0b100 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_xor(dec_insn)),
                _ => None
            },
        0b101 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_srl(dec_insn)),
                0b010_0000 => Some(processor.process_sra(dec_insn)),
                _ => None
            },
        0b110 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_or(dec_insn)),
                _ => None
            },
        0b111 =>
            match dec_insn.funct7 {
                0b000_0000 => Some(processor.process_and(dec_insn)),
                _ => None
            },*/
        _ => None
    }
}

pub fn process_instruction<T: InstructionProcessor>(processor: &mut T, insn_bits: u32) ->
    Option<T::InstructionResult> {

    let opcode: u32 = insn_bits & 0x7f;

    match opcode {
        instruction_format::OPCODE_OP => process_opcode_op(processor, insn_bits),
        _ => None
    }
}

pub struct HartState {
    pub registers: [u32; 32],
    pub pc: u32
}

impl HartState {
    fn new() -> Self {
        HartState {
            registers: [0; 32],
            pc: 0
        }
    }
}

pub trait Memory {
    fn read_mem(&mut self, addr: u32) -> Option<u32>;
}

pub struct InstructionExecutor<'a, M : Memory> {
    mem: &'a mut M,
    hart_state: &'a mut HartState
}

#[derive(Debug, PartialEq)]
pub enum InstructionException {
    // TODO: Better to name the fields?
    IllegalInstruction(u32, u32),
    FetchError(u32)
}

impl<'a, M : Memory> InstructionProcessor for InstructionExecutor<'a, M> {
    type InstructionResult = Result<Option<u32>, InstructionException>;

    fn process_add(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult {
        let result = self.hart_state.registers[dec_insn.rs1] +
            self.hart_state.registers[dec_insn.rs2];

        self.hart_state.registers[dec_insn.rd] = result;

        Ok(None)
    }

    fn process_sub(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult {
        let result = self.hart_state.registers[dec_insn.rs1] -
            self.hart_state.registers[dec_insn.rs2];

        self.hart_state.registers[dec_insn.rd] = result;

        Ok(None)
    }
}

impl<'a, M : Memory> InstructionExecutor<'a, M> {
    fn step(&mut self) -> Result<(), InstructionException> {
        if let Some(next_insn) = self.mem.read_mem(self.hart_state.pc) {
            let step_result = process_instruction(self, next_insn);

            match step_result {
                Some(Ok(None)) => {
                    self.hart_state.pc = self.hart_state.pc + 4;
                    Ok(())
                },
                Some(Ok(Some(next_pc))) => {
                    self.hart_state.pc = next_pc;
                    Ok(())
                }
                Some(Err(e)) => Err(e),
                None => Err(InstructionException::IllegalInstruction(self.hart_state.pc, next_insn))
            }
        } else {
            Err(InstructionException::FetchError(self.hart_state.pc))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::instruction_format;

    #[test]
    fn test_rtype() {
        assert_eq!(instruction_format::RType::new(0x0), instruction_format::RType {
            funct7: 0,
            rs2: 0,
            rs1: 0,
            funct3: 0,
            rd: 0
        })
    }

    #[test]
    fn test_itype() {
        // addi x23, x31, 2047
        assert_eq!(instruction_format::IType::new(0x7fff8b93), instruction_format::IType {
            imm: 2047,
            rs1: 31,
            funct3: 0,
            rd: 23
        });

        // addi x23, x31, -1
        assert_eq!(instruction_format::IType::new(0xffff8b93), instruction_format::IType {
            imm: -1,
            rs1: 31,
            funct3: 0,
            rd: 23
        });

        // addi x23, x31, -2
        assert_eq!(instruction_format::IType::new(0xffef8b93), instruction_format::IType {
            imm: -2,
            rs1: 31,
            funct3: 0,
            rd: 23
        });

        // ori x13, x7, 4
        assert_eq!(instruction_format::IType::new(0x8003e693), instruction_format::IType {
            imm: -2048,
            rs1: 7,
            funct3: 0b110,
            rd: 13
        });
    }

    #[test]
    fn test_stype() {
        // sb x31, -2048(x15)
        assert_eq!(instruction_format::SType::new(0x81f78023), instruction_format::SType {
            imm: -2048,
            rs2: 31,
            rs1: 15,
            funct3: 0,
        });

        // sh x18, 2047(x3)
        assert_eq!(instruction_format::SType::new(0x7f219fa3), instruction_format::SType {
            imm: 2047,
            rs2: 18,
            rs1: 3,
            funct3: 1,
        });

        // sw x8, 1(x23)
        assert_eq!(instruction_format::SType::new(0x008ba0a3), instruction_format::SType {
            imm: 1,
            rs2: 8,
            rs1: 23,
            funct3: 2,
        });

        // sw x5, -1(x25)
        assert_eq!(instruction_format::SType::new(0xfe5cafa3), instruction_format::SType {
            imm: -1,
            rs2: 5,
            rs1: 25,
            funct3: 2,
        });

        // sw x13, 7(x12)
        assert_eq!(instruction_format::SType::new(0x00d623a3), instruction_format::SType {
            imm: 7,
            rs2: 13,
            rs1: 12,
            funct3: 2,
        });

        // sw x13, -7(x12)
        assert_eq!(instruction_format::SType::new(0xfed62ca3), instruction_format::SType {
            imm: -7,
            rs2: 13,
            rs1: 12,
            funct3: 2,
        });
    }

    #[test]
    fn test_btype() {
        // beq x10, x14, .-4096
        assert_eq!(instruction_format::BType::new(0x80e50063), instruction_format::BType {
            imm: -4096,
            rs1: 10,
            rs2: 14,
            funct3: 0b000
        });

        // blt x3, x21, .+4094
        assert_eq!(instruction_format::BType::new(0x7f51cfe3), instruction_format::BType {
            imm: 4094,
            rs1: 3,
            rs2: 21,
            funct3: 0b100
        });

        // bge x18, x0, .-2
        assert_eq!(instruction_format::BType::new(0xfe095fe3), instruction_format::BType {
            imm: -2,
            rs1: 18,
            rs2: 0,
            funct3: 0b101
        });

        // bne x15, x16, .+2
        assert_eq!(instruction_format::BType::new(0x01079163), instruction_format::BType {
            imm: 2,
            rs1: 15,
            rs2: 16,
            funct3: 0b001
        });

        // bgeu x31, x8, .+18
        assert_eq!(instruction_format::BType::new(0x008ff963), instruction_format::BType {
            imm: 18,
            rs1: 31,
            rs2: 8,
            funct3: 0b111
        });

        // bgeu x31, x8, .-18
        assert_eq!(instruction_format::BType::new(0xfe8ff7e3), instruction_format::BType {
            imm: -18,
            rs1: 31,
            rs2: 8,
            funct3: 0b111
        });
    }

    #[test]
    fn test_utype() {
        // lui x0, 0xfffff
        assert_eq!(instruction_format::UType::new(0xfffff037), instruction_format::UType {
            imm: (0xfffff000 as u32) as i32,
            rd: 0,
        });

        // lui x31, 0x0
        assert_eq!(instruction_format::UType::new(0x00000fb7), instruction_format::UType {
            imm: 0x0,
            rd: 31,
        });

        // lui x17, 0x123ab
        assert_eq!(instruction_format::UType::new(0x123ab8b7), instruction_format::UType {
            imm: 0x123ab000,
            rd: 17,
        });
    }

    #[test]
    fn test_jtype() {
        // jal x0, .+0xffffe
        assert_eq!(instruction_format::JType::new(0x7ffff06f), instruction_format::JType {
            imm: 0xffffe,
            rd: 0,
        });

        // jal x31, .-0x100000
        assert_eq!(instruction_format::JType::new(0x80000fef), instruction_format::JType {
            imm: -0x100000,
            rd: 31,
        });

        // jal x13, .-2
        assert_eq!(instruction_format::JType::new(0xfffff6ef), instruction_format::JType {
            imm: -2,
            rd: 13,
        });

        // jal x13, .+2
        assert_eq!(instruction_format::JType::new(0x002006ef), instruction_format::JType {
            imm: 2,
            rd: 13,
        });

        // jal x26, .-46
        assert_eq!(instruction_format::JType::new(0xfd3ffd6f), instruction_format::JType {
            imm: -46,
            rd: 26,
        });

        // jal x26, .+46
        assert_eq!(instruction_format::JType::new(0x02e00d6f), instruction_format::JType {
            imm: 46,
            rd: 26,
        });
    }

    #[test]
    fn test_insn_string_output() {
        let mut outputter = InstructionStringOutputter {};

        let test_insn : u32 = 0x009607b3;

        assert_eq!(process_instruction(&mut outputter, test_insn), Some(String::from("add x15, x12, x9")));
    }

    pub struct TestMemory {
        pub mem: Vec<u32>
    }

    impl TestMemory {
        pub fn new() -> TestMemory {
            TestMemory {
                mem: vec![0x009607b3, 0x0, 0x0, 0x0, 0x0]
            }
        }
    }

    impl Memory for TestMemory {
        fn read_mem(&mut self, addr: u32) -> Option<u32> {
            self.mem.get(addr as usize).copied()
        }
    }

    #[test]
    fn test_insn_execute() {
        let mut hart = HartState::new();
        let mut mem = TestMemory::new();

        hart.registers[12] = 1;
        hart.registers[9] = 2;
        hart.pc = 0;

        // TODO: With the 'executor' concept we need to effectively create a new one each step as
        // it's meant to be just taking a reference to things to execute, but then if we want to
        // access those things we either do it via the executor or create a new one before the next
        // step to allow access via the 'main' object, could just make step part of the 'main'
        // object? Having the executor only coupled to a bare minimum of state could be good?
        let mut executor = InstructionExecutor {
            hart_state : &mut hart,
            mem : &mut mem
        };

        assert_eq!(executor.step(), Ok(()));

        assert_eq!(executor.hart_state.registers[15], 3);

        //assert_eq!(executor.step(), Err(InstructionException::FetchError(4)));
        assert_eq!(executor.step(), Err(InstructionException::IllegalInstruction(4, 0)));
    }
}
