// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use super::CSR;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use std::convert::TryFrom;

// TODO: May be useful to have generic functionality to take a u32 and construct a CSR from it. Can
// manually create CSR then use write to do this, can we do something more generic (e.g. that uses
// write)?

#[derive(PartialEq, Clone, Copy, Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum CSRAddr {
    mvendorid = 0xf11,
    marchid = 0xf12,
    mimpid = 0xf13,
    mhartid = 0xf14,
    mstatus = 0x300,
    misa = 0x301,
    mie = 0x304,
    mtvec = 0x305,
    mcounteren = 0x306,
    mscratch = 0x340,
    mepc = 0x341,
    mcause = 0x342,
    mtval = 0x343,
    mip = 0x344,
    mcycle = 0xb00,
    minstret = 0xb02,
    mhpmcounter3 = 0xb03,
    mhpmcounter4 = 0xb04,
    mhpmcounter5 = 0xb05,
    mhpmcounter6 = 0xb06,
    mhpmcounter7 = 0xb07,
    mhpmcounter8 = 0xb08,
    mhpmcounter9 = 0xb09,
    mhpmcounter10 = 0xb0a,
    mhpmcounter11 = 0xb0b,
    mhpmcounter12 = 0xb0c,
    mhpmcounter13 = 0xb0d,
    mhpmcounter14 = 0xb0e,
    mhpmcounter15 = 0xb0f,
    mhpmcounter16 = 0xb10,
    mhpmcounter17 = 0xb11,
    mhpmcounter18 = 0xb12,
    mhpmcounter19 = 0xb13,
    mhpmcounter20 = 0xb14,
    mhpmcounter21 = 0xb15,
    mhpmcounter22 = 0xb16,
    mhpmcounter23 = 0xb17,
    mhpmcounter24 = 0xb18,
    mhpmcounter25 = 0xb19,
    mhpmcounter26 = 0xb1a,
    mhpmcounter27 = 0xb1b,
    mhpmcounter28 = 0xb1c,
    mhpmcounter29 = 0xb1d,
    mhpmcounter30 = 0xb1e,
    mhpmcounter31 = 0xb1f,
    mcountinhibit = 0x320,
    mhpmevent3 = 0x323,
    mhpmevent4 = 0x324,
    mhpmevent5 = 0x325,
    mhpmevent6 = 0x326,
    mhpmevent7 = 0x327,
    mhpmevent8 = 0x328,
    mhpmevent9 = 0x329,
    mhpmevent10 = 0x32a,
    mhpmevent11 = 0x32b,
    mhpmevent12 = 0x32c,
    mhpmevent13 = 0x32d,
    mhpmevent14 = 0x32e,
    mhpmevent15 = 0x32f,
    mhpmevent16 = 0x330,
    mhpmevent17 = 0x331,
    mhpmevent18 = 0x332,
    mhpmevent19 = 0x333,
    mhpmevent20 = 0x334,
    mhpmevent21 = 0x335,
    mhpmevent22 = 0x336,
    mhpmevent23 = 0x337,
    mhpmevent24 = 0x338,
    mhpmevent25 = 0x339,
    mhpmevent26 = 0x33a,
    mhpmevent27 = 0x33b,
    mhpmevent28 = 0x33c,
    mhpmevent29 = 0x33d,
    mhpmevent30 = 0x33e,
    mhpmevent31 = 0x33f,
    cycle = 0xc00,
}

#[derive(PartialEq, Clone, Copy, Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u32)]
pub enum PrivLevel {
    U = 0,
    S = 1,
    M = 3,
}

#[derive(PartialEq, Clone, Copy, Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u32)]
pub enum MisaMXL {
    XLen32 = 1,
    XLen64 = 2,
    XLen128 = 3,
}

#[derive(Default)]
pub struct Generic {
    pub val: u32,
}

impl CSR for Generic {
    fn read(&self) -> u32 {
        self.val
    }

    fn write(&mut self, val: u32) {
        self.val = val;
    }
}

pub struct MIsa {
    pub mxl: MisaMXL,
    pub i: bool,
    pub m: bool,
}

impl Default for MIsa {
    fn default() -> Self {
        MIsa {
            mxl: MisaMXL::XLen32,
            i: true,
            m: true,
        }
    }
}

impl CSR for MIsa {
    fn read(&self) -> u32 {
        let mut read_data = u32::from(self.mxl) << 30;
        if self.i {
            read_data |= 1 << 8;
        }

        if self.m {
            read_data |= 1 << 12;
        }

        read_data
    }

    fn write(&mut self, _val: u32) {}
}

pub struct MVendorID {
    pub bank: u32,
    pub offset: u32,
}

impl Default for MVendorID {
    fn default() -> Self {
        MVendorID { bank: 0, offset: 0 }
    }
}

impl CSR for MVendorID {
    fn read(&self) -> u32 {
        (self.bank & 0x7f) | ((self.offset & 0x1ffffff) << 7)
    }

    fn write(&mut self, _val: u32) {}
}

pub struct MStatus {
    pub mie: bool,
    pub mpie: bool,
    pub mpp: PrivLevel,
}

impl Default for MStatus {
    fn default() -> Self {
        MStatus {
            mie: false,
            mpie: true,
            mpp: PrivLevel::M,
        }
    }
}

impl CSR for MStatus {
    fn read(&self) -> u32 {
        let mut read_data = (u32::from(self.mpp)) << 11;

        if self.mie {
            read_data |= 1 << 3;
        }

        if self.mpie {
            read_data |= 1 << 7;
        }

        read_data
    }

    fn write(&mut self, val: u32) {
        self.mie = val & (1 << 3) != 0;
        self.mpie = val & (1 << 7) != 0;

        self.mpp = PrivLevel::try_from((val >> 11) & 0x3).unwrap_or(PrivLevel::M);
        if self.mpp != PrivLevel::M {
            self.mpp = PrivLevel::M;
        }
    }
}

pub struct MTVec {
    pub base: u32,
    pub vectored_mode: bool,
}

impl Default for MTVec {
    fn default() -> Self {
        MTVec {
            base: 0,
            vectored_mode: false,
        }
    }
}

impl CSR for MTVec {
    fn read(&self) -> u32 {
        let mut read_data = self.base & 0xfffffffc;

        if self.vectored_mode {
            read_data |= 1;
        }

        read_data
    }

    fn write(&mut self, val: u32) {
        self.base = val & 0xfffffffc;
        self.vectored_mode = (val & 3) == 1;
    }
}

pub struct MIx {
    pub external: bool,
    pub timer: bool,
    pub software: bool,
}

impl Default for MIx {
    fn default() -> Self {
        MIx {
            external: false,
            timer: false,
            software: false,
        }
    }
}

impl CSR for MIx {
    fn read(&self) -> u32 {
        let mut read_data = 0;

        if self.external {
            read_data |= 1 << 11;
        }

        if self.timer {
            read_data |= 1 << 7;
        }

        if self.software {
            read_data |= 1 << 3;
        }

        read_data
    }

    fn write(&mut self, val: u32) {
        self.external = val & (1 << 11) != 0;
        self.timer = val & (1 << 7) != 0;
        self.software = val & (1 << 3) != 0;
    }
}

pub struct MCountInhibit {
    pub cycle: bool,
    pub instret: bool,
}

impl Default for MCountInhibit {
    fn default() -> Self {
        MCountInhibit {
            cycle: false,
            instret: false,
        }
    }
}

impl CSR for MCountInhibit {
    fn read(&self) -> u32 {
        let mut read_data = 0;

        if self.cycle {
            read_data |= 1;
        }

        if self.instret {
            read_data |= 1 << 2;
        }

        read_data
    }

    fn write(&mut self, val: u32) {
        self.cycle = val & 1 != 0;
        self.instret = val & (1 << 2) != 0;
    }
}

#[derive(PartialEq, Clone, Copy, Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u32)]
pub enum ExceptionCause {
    InstructionAddressMisaligned = 0x0,
    InstructionAccessFault = 0x1,
    IllegalInstruction = 0x2,
    Breakpoint = 0x3,
    LoadAddressMisaligned = 0x4,
    LoadAccessFault = 0x5,
    StoreAddressMisaligned = 0x6,
    StoreAccessFault = 0x7,
    ECallMMode = 0xb,
}

pub struct MCause {
    pub cause: u32,
}

impl Default for MCause {
    fn default() -> Self {
        MCause { cause: 0 }
    }
}

impl CSR for MCause {
    fn read(&self) -> u32 {
        self.cause
    }

    fn write(&mut self, val: u32) {
        if val & 0x80000000 != 0 {
            // For interrupt causes accept any value
            self.cause = val;
        } else {
            // For exception interrupt causes only accept defined ones, cause becomes
            // IllegalInstruction for any undefined values.
            self.cause = ExceptionCause::try_from(val)
                .unwrap_or(ExceptionCause::IllegalInstruction)
                .into();
        }
    }
}

#[derive(Default)]
pub struct CSRSet {
    pub misa: MIsa,
    pub mvendorid: MVendorID,
    pub marchid: Generic,
    pub mimpid: Generic,
    pub mhartid: Generic,
    pub mstatus: MStatus,
    pub mtvec: MTVec,
    pub mip: MIx,
    pub mie: MIx,
    pub mcycle: Generic,
    pub minstret: Generic,
    pub mhpmcounter: Generic,
    pub mhpmevent: Generic,
    pub mcounteren: Generic,
    pub mcountinhibit: MCountInhibit,
    pub mepc: Generic,
    pub mcause: MCause,
    pub mtval: Generic,
    pub mscratch: Generic,
}

impl CSRSet {
    pub fn get_csr(&mut self, addr: u32) -> Option<&mut dyn CSR> {
        let csr_addr = CSRAddr::try_from(addr).ok()?;

        if addr >= CSRAddr::mhpmcounter3.into() && addr <= CSRAddr::mhpmcounter31.into() {
            return Some(&mut self.mhpmcounter);
        }

        if addr >= CSRAddr::mhpmevent3.into() && addr <= CSRAddr::mhpmevent31.into() {
            return Some(&mut self.mhpmevent);
        }

        Some(match csr_addr {
            CSRAddr::mvendorid => &mut self.mvendorid,
            CSRAddr::marchid => &mut self.marchid,
            CSRAddr::mimpid => &mut self.mimpid,
            CSRAddr::mhartid => &mut self.mhartid,
            CSRAddr::mstatus => &mut self.mstatus,
            CSRAddr::misa => &mut self.misa,
            CSRAddr::mie => &mut self.mie,
            CSRAddr::mtvec => &mut self.mtvec,
            CSRAddr::mcounteren => &mut self.mcounteren,
            CSRAddr::mscratch => &mut self.mscratch,
            CSRAddr::mepc => &mut self.mepc,
            CSRAddr::mcause => &mut self.mcause,
            CSRAddr::mtval => &mut self.mtval,
            CSRAddr::mip => &mut self.mip,
            CSRAddr::mcycle => &mut self.mcycle,
            CSRAddr::minstret => &mut self.minstret,
            CSRAddr::mcountinhibit => &mut self.mcountinhibit,
            _ => {
                return None;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvendorid() {
        let mut csr_set = CSRSet::default();

        csr_set.mvendorid.bank = 0x5a;
        csr_set.mvendorid.offset = 0xabcd;
        let mvendorid = csr_set.get_csr(CSRAddr::mvendorid.into()).unwrap();
        assert_eq!(mvendorid.read(), 0x55E6DA);
    }

    #[test]
    fn test_misa() {
        let mut csr_set = CSRSet::default();

        let misa = csr_set.get_csr(CSRAddr::misa.into()).unwrap();
        assert_eq!(misa.read(), 0x40001100);
    }

    #[test]
    fn test_mstatus() {
        let mut csr_set = CSRSet::default();

        let mstatus = csr_set.get_csr(CSRAddr::mstatus.into()).unwrap();
        mstatus.write(0xffffffff);
        assert_eq!(mstatus.read(), 0x1888);
        assert!(csr_set.mstatus.mie);
        assert!(csr_set.mstatus.mpie);
        assert_eq!(csr_set.mstatus.mpp, PrivLevel::M);

        let mstatus = csr_set.get_csr(CSRAddr::mstatus.into()).unwrap();
        mstatus.write(0x00000000);
        assert!(!csr_set.mstatus.mie);
        assert!(!csr_set.mstatus.mpie);
        assert_eq!(csr_set.mstatus.mpp, PrivLevel::M);
    }

    #[test]
    fn test_mcause() {
        let mut csr_set = CSRSet::default();

        let mcause = csr_set.get_csr(CSRAddr::mcause.into()).unwrap();
        assert_eq!(mcause.read(), 0);
        mcause.write(ExceptionCause::ECallMMode.into());
        assert_eq!(mcause.read(), ExceptionCause::ECallMMode.into());
        mcause.write(0xffffffff);
        assert_eq!(mcause.read(), 0xffffffff);
        mcause.write(0x7fffffff);
        assert_eq!(mcause.read(), ExceptionCause::IllegalInstruction.into());
    }

    #[test]
    fn test_mtvec() {
        let mut csr_set = CSRSet::default();

        let mtvec = csr_set.get_csr(CSRAddr::mtvec.into()).unwrap();
        mtvec.write(0xffffffff);
        assert_eq!(mtvec.read(), 0xfffffffc);
        assert!(!csr_set.mtvec.vectored_mode);
        assert_eq!(csr_set.mtvec.base, 0xfffffffc);
        let mtvec = csr_set.get_csr(CSRAddr::mtvec.into()).unwrap();
        mtvec.write(0xfffffffd);
        assert_eq!(mtvec.read(), 0xfffffffd);
        assert!(csr_set.mtvec.vectored_mode);
        assert_eq!(csr_set.mtvec.base, 0xfffffffc);
    }

    #[test]
    fn test_mix() {
        let mut csr_set = CSRSet::default();

        let mie = csr_set.get_csr(CSRAddr::mie.into()).unwrap();
        mie.write(0xffffffff);
        assert_eq!(mie.read(), 0x00000888);
        assert!(csr_set.mie.external);
        assert!(csr_set.mie.timer);
        assert!(csr_set.mie.software);
        let mie = csr_set.get_csr(CSRAddr::mie.into()).unwrap();
        mie.write(0x00000000);
        assert_eq!(mie.read(), 0x000000000);
        assert!(!csr_set.mie.external);
        assert!(!csr_set.mie.timer);
        assert!(!csr_set.mie.software);
    }

    #[test]
    fn test_mcountinhibit() {
        let mut csr_set = CSRSet::default();

        let mcountinhibit = csr_set.get_csr(CSRAddr::mcountinhibit.into()).unwrap();
        mcountinhibit.write(0xffffffff);
        assert_eq!(mcountinhibit.read(), 0x00000005);
        assert!(csr_set.mcountinhibit.cycle);
        assert!(csr_set.mcountinhibit.instret);
        let mcountinhibit = csr_set.get_csr(CSRAddr::mcountinhibit.into()).unwrap();
        mcountinhibit.write(0x00000000);
        assert_eq!(mcountinhibit.read(), 0x00000000);
        assert!(!csr_set.mcountinhibit.cycle);
        assert!(!csr_set.mcountinhibit.instret);
    }

    #[test]
    fn test_invalid_csr() {
        let mut csr_set = CSRSet::default();

        assert!(csr_set.get_csr(0xfff).is_none());
    }
}
