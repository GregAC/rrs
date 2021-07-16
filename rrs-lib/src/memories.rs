use super::{MemAccessSize, Memory};

pub struct VecMemory {
    pub mem: Vec<u32>,
}

impl VecMemory {
    pub fn new(init_mem: Vec<u32>) -> VecMemory {
        VecMemory { mem: init_mem }
    }
}

impl Memory for VecMemory {
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

        Some((read_data >> (shift * 8)) & mask)
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

        let write_mask = !(mask << (shift * 8));

        let word_addr = (addr >> 2) as usize;

        if let Some(update_data) = self.mem.get(word_addr) {
            let new = (update_data & write_mask) | ((store_data & mask) << (shift * 8));
            self.mem[word_addr] = new;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_vec_memory() {
        let mut test_mem = VecMemory::new(vec![0xdeadbeef, 0xbaadf00d]);

        assert_eq!(test_mem.read_mem(0x0, MemAccessSize::Byte), Some(0xef));

        assert_eq!(test_mem.read_mem(0x5, MemAccessSize::Byte), Some(0xf0));

        assert_eq!(
            test_mem.read_mem(0x6, MemAccessSize::HalfWord),
            Some(0xbaad)
        );

        assert_eq!(
            test_mem.read_mem(0x4, MemAccessSize::Word),
            Some(0xbaadf00d)
        );

        assert_eq!(test_mem.write_mem(0x7, MemAccessSize::Byte, 0xff), true);

        assert_eq!(
            test_mem.write_mem(0x2, MemAccessSize::HalfWord, 0xaaaaface),
            true
        );

        assert_eq!(
            test_mem.write_mem(0x1, MemAccessSize::Byte, 0x1234abcd),
            true
        );

        assert_eq!(
            test_mem.read_mem(0x0, MemAccessSize::Word),
            Some(0xfacecdef)
        );

        assert_eq!(
            test_mem.read_mem(0x4, MemAccessSize::Word),
            Some(0xffadf00d)
        );
    }
}
