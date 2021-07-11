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
        pub rd: usize,
    }

    impl RType {
        pub fn new(insn: u32) -> RType {
            RType {
                funct7: (insn >> 25) & 0x7f,
                rs2: ((insn >> 20) & 0x1f) as usize,
                rs1: ((insn >> 15) & 0x1f) as usize,
                funct3: (insn >> 12) & 0x7,
                rd: ((insn >> 7) & 0x1f) as usize,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct IType {
        pub imm: i32,
        pub rs1: usize,
        pub funct3: u32,
        pub rd: usize,
    }

    impl IType {
        pub fn new(insn: u32) -> IType {
            let uimm: i32 = ((insn >> 20) & 0x7ff) as i32;

            let imm: i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 11)
            } else {
                uimm
            };

            IType {
                imm,
                rs1: ((insn >> 15) & 0x1f) as usize,
                funct3: (insn >> 12) & 0x7,
                rd: ((insn >> 7) & 0x1f) as usize,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct ITypeShamt {
        pub funct7: u32,
        pub shamt: u32,
        pub rs1: usize,
        pub funct3: u32,
        pub rd: usize,
    }

    impl ITypeShamt {
        pub fn new(insn: u32) -> ITypeShamt {
            let itype = IType::new(insn);
            let shamt = (itype.imm as u32) & 0x1f;

            ITypeShamt {
                funct7: (insn >> 25) & 0x7f,
                shamt,
                rs1: itype.rs1,
                funct3: itype.funct3,
                rd: itype.rd,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct SType {
        pub imm: i32,
        pub rs2: usize,
        pub rs1: usize,
        pub funct3: u32,
    }

    impl SType {
        pub fn new(insn: u32) -> SType {
            let uimm: i32 = (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1f)) as i32;

            let imm: i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 11)
            } else {
                uimm
            };

            SType {
                imm: imm,
                rs2: ((insn >> 20) & 0x1f) as usize,
                rs1: ((insn >> 15) & 0x1f) as usize,
                funct3: (insn >> 12) & 0x7,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct BType {
        pub imm: i32,
        pub rs2: usize,
        pub rs1: usize,
        pub funct3: u32,
    }

    impl BType {
        pub fn new(insn: u32) -> BType {
            let uimm: i32 =
                (((insn >> 20) & 0x7e0) | ((insn >> 7) & 0x1e) | ((insn & 0x80) << 4)) as i32;

            let imm: i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 12)
            } else {
                uimm
            };

            println!("imm: {}", imm);

            BType {
                imm: imm,
                rs2: ((insn >> 20) & 0x1f) as usize,
                rs1: ((insn >> 15) & 0x1f) as usize,
                funct3: (insn >> 12) & 0x7,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct UType {
        pub imm: i32,
        pub rd: usize,
    }

    impl UType {
        pub fn new(insn: u32) -> UType {
            UType {
                imm: (insn & 0xffff_f000) as i32,
                rd: ((insn >> 7) & 0x1f) as usize,
            }
        }
    }

    #[derive(Debug, PartialEq)]
    pub struct JType {
        pub imm: i32,
        pub rd: usize,
    }

    impl JType {
        pub fn new(insn: u32) -> JType {
            let uimm: i32 =
                ((insn & 0xff000) | ((insn & 0x100000) >> 9) | ((insn >> 20) & 0x7fe)) as i32;

            let imm: i32 = if (insn & 0x8000_0000) != 0 {
                uimm - (1 << 20)
            } else {
                uimm
            };

            JType {
                imm: imm,
                rd: ((insn >> 7) & 0x1f) as usize,
            }
        }
    }
}

pub trait InstructionProcessor {
    type InstructionResult;

    fn process_add(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_sub(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_sll(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_slt(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_sltu(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_xor(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_srl(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_sra(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_or(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;
    fn process_and(&mut self, dec_insn: instruction_format::RType) -> Self::InstructionResult;

    fn process_addi(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_slli(&mut self, dec_insn: instruction_format::ITypeShamt)
        -> Self::InstructionResult;
    fn process_slti(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_sltui(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_xori(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_srli(&mut self, dec_insn: instruction_format::ITypeShamt)
        -> Self::InstructionResult;
    fn process_srai(&mut self, dec_insn: instruction_format::ITypeShamt)
        -> Self::InstructionResult;
    fn process_ori(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_andi(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;

    fn process_lui(&mut self, dec_insn: instruction_format::UType) -> Self::InstructionResult;
    fn process_auipc(&mut self, dec_insn: instruction_format::UType) -> Self::InstructionResult;

    fn process_beq(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult;
    fn process_bne(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult;
    fn process_blt(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult;
    fn process_bltu(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult;
    fn process_bge(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult;
    fn process_bgeu(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult;

    fn process_lb(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_lbu(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_lh(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_lhu(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
    fn process_lw(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;

    fn process_sb(&mut self, dec_insn: instruction_format::SType) -> Self::InstructionResult;
    fn process_sh(&mut self, dec_insn: instruction_format::SType) -> Self::InstructionResult;
    fn process_sw(&mut self, dec_insn: instruction_format::SType) -> Self::InstructionResult;

    fn process_jal(&mut self, dec_insn: instruction_format::JType) -> Self::InstructionResult;
    fn process_jalr(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult;
}

pub struct InstructionStringOutputter {
    pub insn_pc: u32,
}

use paste::paste;

// TODO: Implement display for instruction formats then make generic string out macro that takes an
// instruction format type, have generic function for actual output? Macro just a convenience for
// avoiding too much boiler plate?
macro_rules! string_out_for_alu_reg_op {
    ($name:ident) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_format::RType
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
                dec_insn: instruction_format::IType
            ) -> Self::InstructionResult {
                format!("{} x{}, x{}, x{}", stringify!($name), dec_insn.rd, dec_insn.rs1,
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
                dec_insn: instruction_format::ITypeShamt
            ) -> Self::InstructionResult {
                format!("{} x{}, x{}, x{}", stringify!($name), dec_insn.rd, dec_insn.rs1,
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
                    dec_insn: instruction_format::BType
                ) -> Self::InstructionResult {
                    let branch_pc = self.insn_pc.wrapping_add(dec_insn.imm as u32);

                    format!("{} x{}, x{}, {:08x}", stringify!($name), dec_insn.rs1, dec_insn.rs2,
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
                    dec_insn: instruction_format::IType
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
                    dec_insn: instruction_format::SType
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
    string_out_for_alu_ops! {add, slt, sltu, xor, or, and}
    string_out_for_alu_reg_op! {sub}
    string_out_for_shift_ops! {sll, srl, sra}

    fn process_lui(&mut self, dec_insn: instruction_format::UType) -> Self::InstructionResult {
        let shifted_imm = (dec_insn.imm as u32) << 12;
        format!("lui x{}, 0x{:08x}", dec_insn.rd, shifted_imm)
    }

    fn process_auipc(&mut self, dec_insn: instruction_format::UType) -> Self::InstructionResult {
        let final_imm = self.insn_pc.wrapping_add(dec_insn.imm as u32);
        format!("auipc x{}, 0x{:08x}", dec_insn.rd, final_imm)
    }

    string_out_for_branch_ops! {beq, bne, bge, bgeu, blt, bltu}
    string_out_for_load_ops! {lb, lbu, lh, lhu, lw}
    string_out_for_store_ops! {sb, sh, sw}

    fn process_jal(&mut self, dec_insn: instruction_format::JType) -> Self::InstructionResult {
        let target_pc = self.insn_pc.wrapping_add(dec_insn.imm as u32);
        format!("jal x{}, 0x{:08x}", dec_insn.rd, target_pc)
    }

    fn process_jalr(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult {
        format!("jalr x{}, x{}, {}", dec_insn.rd, dec_insn.rs1, dec_insn.imm)
    }
}

fn process_opcode_op<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_format::RType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_add(dec_insn)),
            0b010_0000 => Some(processor.process_sub(dec_insn)),
            _ => None,
        },
        0b001 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_sll(dec_insn)),
            _ => None,
        },
        0b010 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_slt(dec_insn)),
            _ => None,
        },
        0b011 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_sltu(dec_insn)),
            _ => None,
        },
        0b100 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_xor(dec_insn)),
            _ => None,
        },
        0b101 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_srl(dec_insn)),
            0b010_0000 => Some(processor.process_sra(dec_insn)),
            _ => None,
        },
        0b110 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_or(dec_insn)),
            _ => None,
        },
        0b111 => match dec_insn.funct7 {
            0b000_0000 => Some(processor.process_and(dec_insn)),
            _ => None,
        },
        _ => None,
    }
}

fn process_opcode_op_imm<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_format::IType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_addi(dec_insn)),
        0b001 => Some(processor.process_slli(instruction_format::ITypeShamt::new(insn_bits))),
        0b010 => Some(processor.process_slti(dec_insn)),
        0b011 => Some(processor.process_sltui(dec_insn)),
        0b100 => Some(processor.process_xori(dec_insn)),
        0b101 => {
            let dec_insn_shamt = instruction_format::ITypeShamt::new(insn_bits);
            match dec_insn_shamt.funct7 {
                0b000_0000 => Some(processor.process_srli(dec_insn_shamt)),
                0b010_0000 => Some(processor.process_srai(dec_insn_shamt)),
                _ => None,
            }
        }
        0b110 => Some(processor.process_ori(dec_insn)),
        0b111 => Some(processor.process_andi(dec_insn)),
        _ => None,
    }
}

fn process_opcode_branch<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_format::BType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_beq(dec_insn)),
        0b001 => Some(processor.process_bne(dec_insn)),
        0b100 => Some(processor.process_blt(dec_insn)),
        0b101 => Some(processor.process_bge(dec_insn)),
        0b110 => Some(processor.process_bltu(dec_insn)),
        0b111 => Some(processor.process_bgeu(dec_insn)),
        _ => None,
    }
}

fn process_opcode_load<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_format::IType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_lb(dec_insn)),
        0b001 => Some(processor.process_lh(dec_insn)),
        0b010 => Some(processor.process_lw(dec_insn)),
        0b100 => Some(processor.process_lbu(dec_insn)),
        0b101 => Some(processor.process_lhu(dec_insn)),
        _ => None,
    }
}

fn process_opcode_store<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let dec_insn = instruction_format::SType::new(insn_bits);

    match dec_insn.funct3 {
        0b000 => Some(processor.process_sb(dec_insn)),
        0b001 => Some(processor.process_sh(dec_insn)),
        0b010 => Some(processor.process_sw(dec_insn)),
        _ => None,
    }
}

pub fn process_instruction<T: InstructionProcessor>(
    processor: &mut T,
    insn_bits: u32,
) -> Option<T::InstructionResult> {
    let opcode: u32 = insn_bits & 0x7f;

    match opcode {
        instruction_format::OPCODE_OP => process_opcode_op(processor, insn_bits),
        instruction_format::OPCODE_OP_IMM => process_opcode_op_imm(processor, insn_bits),
        instruction_format::OPCODE_LUI => {
            Some(processor.process_lui(instruction_format::UType::new(insn_bits)))
        }
        instruction_format::OPCODE_AUIPC => {
            Some(processor.process_auipc(instruction_format::UType::new(insn_bits)))
        }
        instruction_format::OPCODE_BRANCH => process_opcode_branch(processor, insn_bits),
        instruction_format::OPCODE_LOAD => process_opcode_load(processor, insn_bits),
        instruction_format::OPCODE_STORE => process_opcode_store(processor, insn_bits),
        instruction_format::OPCODE_JAL => {
            Some(processor.process_jal(instruction_format::JType::new(insn_bits)))
        }
        instruction_format::OPCODE_JALR => {
            Some(processor.process_jalr(instruction_format::IType::new(insn_bits)))
        }
        _ => None,
    }
}

pub struct HartState {
    pub registers: [u32; 32],
    pub pc: u32,
}

impl HartState {
    fn new() -> Self {
        HartState {
            registers: [0; 32],
            pc: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub enum MemAccessSize {
    Byte,     // 8 bits
    HalfWord, // 16 bits
    Word,     // 32 bits
}

pub trait Memory {
    fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32>;
    fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool;
}

#[derive(Debug, PartialEq)]
pub enum InstructionException {
    // TODO: Better to name the fields?
    IllegalInstruction(u32, u32),
    FetchError(u32),
    LoadAccessFault(u32),
    StoreAccessFault(u32),
    AlignmentFault(u32),
}

pub struct InstructionExecutor<'a, M: Memory> {
    mem: &'a mut M,
    hart_state: &'a mut HartState,
}

impl<'a, M: Memory> InstructionExecutor<'a, M> {
    fn execute_reg_reg_op<F>(&mut self, dec_insn: instruction_format::RType, op: F)
    where
        F: Fn(u32, u32) -> u32,
    {
        let a = self.hart_state.registers[dec_insn.rs1];
        let b = self.hart_state.registers[dec_insn.rs2];
        let result = op(a, b);
        self.hart_state.registers[dec_insn.rd] = result;
    }

    fn execute_reg_imm_op<F>(&mut self, dec_insn: instruction_format::IType, op: F)
    where
        F: Fn(u32, u32) -> u32,
    {
        let a = self.hart_state.registers[dec_insn.rs1];
        let b = dec_insn.imm as u32;
        let result = op(a, b);
        self.hart_state.registers[dec_insn.rd] = result;
    }

    fn execute_reg_imm_shamt_op<F>(&mut self, dec_insn: instruction_format::ITypeShamt, op: F)
    where
        F: Fn(u32, u32) -> u32,
    {
        let a = self.hart_state.registers[dec_insn.rs1];
        let result = op(a, dec_insn.shamt);
        self.hart_state.registers[dec_insn.rd] = result;
    }

    fn execute_branch<F>(&mut self, dec_insn: instruction_format::BType, cond: F) -> bool
    where
        F: Fn(u32, u32) -> bool,
    {
        let a = self.hart_state.registers[dec_insn.rs1];
        let b = self.hart_state.registers[dec_insn.rs1];

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
        dec_insn: instruction_format::IType,
        size: MemAccessSize,
        signed: bool,
    ) -> Result<(), InstructionException> {
        let addr = self.hart_state.registers[dec_insn.rs1].wrapping_add(dec_insn.imm as u32);

        let align_mask = match size {
            MemAccessSize::Byte => 0x0,
            MemAccessSize::HalfWord => 0x1,
            MemAccessSize::Word => 0x3,
        };

        if (addr & align_mask) != 0x0 {
            return Err(InstructionException::AlignmentFault(addr));
        }

        let mut load_data = match self.mem.read_mem(addr, size) {
            Some(d) => d,
            None => {
                return Err(InstructionException::LoadAccessFault(addr));
            }
        };

        if signed {
            load_data = (match size {
                MemAccessSize::Byte => (load_data as i8) as i32,
                MemAccessSize::HalfWord => (load_data as i16) as i32,
                MemAccessSize::Word => load_data as i32,
            }) as u32;
        }

        self.hart_state.registers[dec_insn.rd] = load_data;
        Ok(())
    }

    fn execute_store(
        &mut self,
        dec_insn: instruction_format::SType,
        size: MemAccessSize,
    ) -> Result<(), InstructionException> {
        let addr = self.hart_state.registers[dec_insn.rs1].wrapping_add(dec_insn.imm as u32);
        let data = self.hart_state.registers[dec_insn.rs2];

        let align_mask = match size {
            MemAccessSize::Byte => 0x0,
            MemAccessSize::HalfWord => 0x1,
            MemAccessSize::Word => 0x3,
        };

        if (addr & align_mask) != 0x0 {
            return Err(InstructionException::AlignmentFault(addr));
        }

        if self.mem.write_mem(addr, size, data) {
            Ok(())
        } else {
            Err(InstructionException::StoreAccessFault(addr))
        }
    }
}

macro_rules! make_alu_op_reg_fn {
    ($name:ident, $op_fn:expr) => {
        paste! {
            fn [<process_ $name>](
                &mut self,
                dec_insn: instruction_format::RType
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
                dec_insn: instruction_format::IType
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
                dec_insn: instruction_format::ITypeShamt
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

    fn process_lui(&mut self, dec_insn: instruction_format::UType) -> Self::InstructionResult {
        let result = (dec_insn.imm as u32) << 12;
        self.hart_state.registers[dec_insn.rd] = result;

        Ok(false)
    }

    fn process_auipc(&mut self, dec_insn: instruction_format::UType) -> Self::InstructionResult {
        let result = self.hart_state.pc.wrapping_add(dec_insn.imm as u32);
        self.hart_state.registers[dec_insn.rd] = result;

        Ok(false)
    }

    fn process_beq(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a == b))
    }

    fn process_bne(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a == b))
    }

    fn process_blt(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| (a as i32) < (b as i32)))
    }

    fn process_bltu(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a < b))
    }

    fn process_bge(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| (a as i32) >= (b as i32)))
    }

    fn process_bgeu(&mut self, dec_insn: instruction_format::BType) -> Self::InstructionResult {
        Ok(self.execute_branch(dec_insn, |a, b| a >= b))
    }

    fn process_lb(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::Byte, true)?;

        Ok(false)
    }

    fn process_lbu(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::Byte, false)?;

        Ok(false)
    }

    fn process_lh(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::HalfWord, true)?;

        Ok(false)
    }

    fn process_lhu(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::HalfWord, false)?;

        Ok(false)
    }

    fn process_lw(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult {
        self.execute_load(dec_insn, MemAccessSize::Word, false)?;

        Ok(false)
    }

    fn process_sb(&mut self, dec_insn: instruction_format::SType) -> Self::InstructionResult {
        self.execute_store(dec_insn, MemAccessSize::Byte)?;

        Ok(false)
    }

    fn process_sh(&mut self, dec_insn: instruction_format::SType) -> Self::InstructionResult {
        self.execute_store(dec_insn, MemAccessSize::HalfWord)?;

        Ok(false)
    }

    fn process_sw(&mut self, dec_insn: instruction_format::SType) -> Self::InstructionResult {
        self.execute_store(dec_insn, MemAccessSize::Word)?;

        Ok(false)
    }

    fn process_jal(&mut self, dec_insn: instruction_format::JType) -> Self::InstructionResult {
        let target_pc = self.hart_state.pc.wrapping_add(dec_insn.imm as u32);
        self.hart_state.registers[dec_insn.rd] = self.hart_state.pc + 4;
        self.hart_state.pc = target_pc;

        Ok(true)
    }

    fn process_jalr(&mut self, dec_insn: instruction_format::IType) -> Self::InstructionResult {
        let mut target_pc =
            self.hart_state.registers[dec_insn.rs1].wrapping_add(dec_insn.imm as u32);
        target_pc &= 0xfffffffe;

        self.hart_state.registers[dec_insn.rd] = self.hart_state.pc + 4;
        self.hart_state.pc = target_pc;

        Ok(true)
    }
}

impl<'a, M: Memory> InstructionExecutor<'a, M> {
    fn step(&mut self) -> Result<(), InstructionException> {
        if let Some(next_insn) = self.mem.read_mem(self.hart_state.pc, MemAccessSize::Word) {
            let step_result = process_instruction(self, next_insn);

            match step_result {
                Some(Ok(pc_updated)) => {
                    if !pc_updated {
                        self.hart_state.pc = self.hart_state.pc + 4;
                    }
                    Ok(())
                }
                Some(Err(e)) => Err(e),
                None => Err(InstructionException::IllegalInstruction(
                    self.hart_state.pc,
                    next_insn,
                )),
            }
        } else {
            Err(InstructionException::FetchError(self.hart_state.pc))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::instruction_format;
    use super::*;

    #[test]
    fn test_rtype() {
        assert_eq!(
            instruction_format::RType::new(0x0),
            instruction_format::RType {
                funct7: 0,
                rs2: 0,
                rs1: 0,
                funct3: 0,
                rd: 0
            }
        )
    }

    #[test]
    fn test_itype() {
        // addi x23, x31, 2047
        assert_eq!(
            instruction_format::IType::new(0x7fff8b93),
            instruction_format::IType {
                imm: 2047,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // addi x23, x31, -1
        assert_eq!(
            instruction_format::IType::new(0xffff8b93),
            instruction_format::IType {
                imm: -1,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // addi x23, x31, -2
        assert_eq!(
            instruction_format::IType::new(0xffef8b93),
            instruction_format::IType {
                imm: -2,
                rs1: 31,
                funct3: 0,
                rd: 23
            }
        );

        // ori x13, x7, 4
        assert_eq!(
            instruction_format::IType::new(0x8003e693),
            instruction_format::IType {
                imm: -2048,
                rs1: 7,
                funct3: 0b110,
                rd: 13
            }
        );
    }

    #[test]
    fn test_itype_shamt() {
        // slli x12, x5, 13
        assert_eq!(
            instruction_format::ITypeShamt::new(0x00d29613),
            instruction_format::ITypeShamt {
                funct7: 0,
                shamt: 13,
                rs1: 5,
                funct3: 0b001,
                rd: 12
            }
        );

        // srli x30, x19, 31
        assert_eq!(
            instruction_format::ITypeShamt::new(0x01f9df13),
            instruction_format::ITypeShamt {
                funct7: 0,
                shamt: 31,
                rs1: 19,
                funct3: 0b101,
                rd: 30
            }
        );

        // srai x7, x23, 0
        assert_eq!(
            instruction_format::ITypeShamt::new(0x400bd393),
            instruction_format::ITypeShamt {
                funct7: 0b0100000,
                shamt: 0,
                rs1: 23,
                funct3: 0b101,
                rd: 7
            }
        );
    }

    #[test]
    fn test_stype() {
        // sb x31, -2048(x15)
        assert_eq!(
            instruction_format::SType::new(0x81f78023),
            instruction_format::SType {
                imm: -2048,
                rs2: 31,
                rs1: 15,
                funct3: 0,
            }
        );

        // sh x18, 2047(x3)
        assert_eq!(
            instruction_format::SType::new(0x7f219fa3),
            instruction_format::SType {
                imm: 2047,
                rs2: 18,
                rs1: 3,
                funct3: 1,
            }
        );

        // sw x8, 1(x23)
        assert_eq!(
            instruction_format::SType::new(0x008ba0a3),
            instruction_format::SType {
                imm: 1,
                rs2: 8,
                rs1: 23,
                funct3: 2,
            }
        );

        // sw x5, -1(x25)
        assert_eq!(
            instruction_format::SType::new(0xfe5cafa3),
            instruction_format::SType {
                imm: -1,
                rs2: 5,
                rs1: 25,
                funct3: 2,
            }
        );

        // sw x13, 7(x12)
        assert_eq!(
            instruction_format::SType::new(0x00d623a3),
            instruction_format::SType {
                imm: 7,
                rs2: 13,
                rs1: 12,
                funct3: 2,
            }
        );

        // sw x13, -7(x12)
        assert_eq!(
            instruction_format::SType::new(0xfed62ca3),
            instruction_format::SType {
                imm: -7,
                rs2: 13,
                rs1: 12,
                funct3: 2,
            }
        );
    }

    #[test]
    fn test_btype() {
        // beq x10, x14, .-4096
        assert_eq!(
            instruction_format::BType::new(0x80e50063),
            instruction_format::BType {
                imm: -4096,
                rs1: 10,
                rs2: 14,
                funct3: 0b000
            }
        );

        // blt x3, x21, .+4094
        assert_eq!(
            instruction_format::BType::new(0x7f51cfe3),
            instruction_format::BType {
                imm: 4094,
                rs1: 3,
                rs2: 21,
                funct3: 0b100
            }
        );

        // bge x18, x0, .-2
        assert_eq!(
            instruction_format::BType::new(0xfe095fe3),
            instruction_format::BType {
                imm: -2,
                rs1: 18,
                rs2: 0,
                funct3: 0b101
            }
        );

        // bne x15, x16, .+2
        assert_eq!(
            instruction_format::BType::new(0x01079163),
            instruction_format::BType {
                imm: 2,
                rs1: 15,
                rs2: 16,
                funct3: 0b001
            }
        );

        // bgeu x31, x8, .+18
        assert_eq!(
            instruction_format::BType::new(0x008ff963),
            instruction_format::BType {
                imm: 18,
                rs1: 31,
                rs2: 8,
                funct3: 0b111
            }
        );

        // bgeu x31, x8, .-18
        assert_eq!(
            instruction_format::BType::new(0xfe8ff7e3),
            instruction_format::BType {
                imm: -18,
                rs1: 31,
                rs2: 8,
                funct3: 0b111
            }
        );
    }

    #[test]
    fn test_utype() {
        // lui x0, 0xfffff
        assert_eq!(
            instruction_format::UType::new(0xfffff037),
            instruction_format::UType {
                imm: (0xfffff000 as u32) as i32,
                rd: 0,
            }
        );

        // lui x31, 0x0
        assert_eq!(
            instruction_format::UType::new(0x00000fb7),
            instruction_format::UType { imm: 0x0, rd: 31 }
        );

        // lui x17, 0x123ab
        assert_eq!(
            instruction_format::UType::new(0x123ab8b7),
            instruction_format::UType {
                imm: 0x123ab000,
                rd: 17,
            }
        );
    }

    #[test]
    fn test_jtype() {
        // jal x0, .+0xffffe
        assert_eq!(
            instruction_format::JType::new(0x7ffff06f),
            instruction_format::JType {
                imm: 0xffffe,
                rd: 0,
            }
        );

        // jal x31, .-0x100000
        assert_eq!(
            instruction_format::JType::new(0x80000fef),
            instruction_format::JType {
                imm: -0x100000,
                rd: 31,
            }
        );

        // jal x13, .-2
        assert_eq!(
            instruction_format::JType::new(0xfffff6ef),
            instruction_format::JType { imm: -2, rd: 13 }
        );

        // jal x13, .+2
        assert_eq!(
            instruction_format::JType::new(0x002006ef),
            instruction_format::JType { imm: 2, rd: 13 }
        );

        // jal x26, .-46
        assert_eq!(
            instruction_format::JType::new(0xfd3ffd6f),
            instruction_format::JType { imm: -46, rd: 26 }
        );

        // jal x26, .+46
        assert_eq!(
            instruction_format::JType::new(0x02e00d6f),
            instruction_format::JType { imm: 46, rd: 26 }
        );
    }

    #[test]
    fn test_insn_string_output() {
        let mut outputter = InstructionStringOutputter { insn_pc: 0 };

        let test_insn: u32 = 0x009607b3;

        assert_eq!(
            process_instruction(&mut outputter, test_insn),
            Some(String::from("add x15, x12, x9"))
        );
    }

    pub struct TestMemory {
        pub mem: Vec<u32>,
    }

    impl TestMemory {
        pub fn new() -> TestMemory {
            TestMemory {
                mem: vec![0x009607b3, 0x0, 0x0, 0x0, 0x0],
            }
        }
    }

    impl Memory for TestMemory {
        fn read_mem(&mut self, addr: u32, size: MemAccessSize) -> Option<u32> {
            let (shift, mask) = match size {
                MemAccessSize::Byte => (addr & 0x3, 0xff),
                MemAccessSize::HalfWord => (addr & 0x2, 0xffff),
                MemAccessSize::Word => (0, 0xffffffff),
            };

            if (addr & 0x3) != shift {
                panic!("Memory read must be aligned");
            }

            let word_addr = addr >> 2;

            let read_data = self.mem.get(word_addr as usize).copied()?;

            Some((read_data >> shift) & mask)
        }

        fn write_mem(&mut self, addr: u32, size: MemAccessSize, store_data: u32) -> bool {
            let (shift, mask) = match size {
                MemAccessSize::Byte => (addr & 0x3, 0xff),
                MemAccessSize::HalfWord => (addr & 0x2, 0xffff),
                MemAccessSize::Word => (0, 0xffffffff),
            };

            if (addr & 0x3) != shift {
                panic!("Memory write must be aligned");
            }

            let write_mask = !(mask << shift);

            let word_addr = addr >> 2;

            if let Some(update_data) = self.mem.get(word_addr as usize) {
                self.mem[word_addr as usize] = (update_data & write_mask) | (store_data << shift);
                true
            } else {
                false
            }
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
            hart_state: &mut hart,
            mem: &mut mem,
        };

        assert_eq!(executor.step(), Ok(()));

        assert_eq!(executor.hart_state.registers[15], 3);

        //assert_eq!(executor.step(), Err(InstructionException::FetchError(4)));
        assert_eq!(
            executor.step(),
            Err(InstructionException::IllegalInstruction(4, 0))
        );
    }
}
