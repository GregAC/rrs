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
        pub imm: u32,
        pub rs1: u32,
        pub funct3: u32,
        pub rd: u32
    }

    impl IType {
        pub fn new(insn: u32) -> IType {
            IType {
                imm: (insn >> 20) & 0xfff,
                rs1: (insn >> 15) & 0x1f,
                funct3: (insn >> 12) & 0x7,
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
    IllegalInstruction,
    FetchError
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
                None => Err(InstructionException::IllegalInstruction)
            }
        } else {
            Err(InstructionException::FetchError)
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
                mem: vec![0x009607b3]
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

        let mut executor = InstructionExecutor {
            hart_state : &mut hart,
            mem : &mut mem
        };

        assert_eq!(executor.step(), Ok(()));

        assert_eq!(hart.registers[15], 3);

        let mut executor = InstructionExecutor {
            hart_state : &mut hart,
            mem : &mut mem
        };

        assert_eq!(executor.step(), Err(InstructionException::FetchError));
    }
}
