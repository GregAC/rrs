// Copyright 2021 Gregory Chadwick <mail@gregchadwick.co.uk>
// Licensed under the Apache License Version 2.0, with LLVM Exceptions, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

use clap::{clap_app, ArgMatches};
use rrs_lib::instruction_executor::{InstructionExecutor, InstructionTrap};
use rrs_lib::instruction_string_outputter::InstructionStringOutputter;
use rrs_lib::memories;
use rrs_lib::memories::{MemorySpace, VecMemory};
use rrs_lib::{HartState, MemAccessSize, Memory};
use std::fs::File;
use std::io;
use std::io::{BufReader, BufWriter, Write};
use std::time::Instant;

#[derive(Default)]
struct CliOpts {
    binary_file: String,
    mem_size: u32,
    mem_base: u32,
    load_addr: u32,
    start_addr: u32,
    log_filename: Option<String>,
    char_out_filename: Option<String>,
}

fn get_arg_matches() -> ArgMatches<'static> {
    clap_app!(("Rust RISC-V Simulator CLI (rrs-cli)") =>
        (version: "0.1")
        (author: "Greg Chadwick <mail@gregchadwick.co.uk>")
        (about: "A RISC-V Instruction Set Simulator (ISS)")
        (@arg binary_file: -b --binary <file> "Binary file to load")
        (@arg mem_size: --mem_size [size] "Size of simulated memory in bytes")
        (@arg mem_base: --mem_base [addr] "Base address of simulated memory")
        (@arg load_addr: --load_addr [addr] "Address to load the binary at")
        (@arg start_addr: --start_addr [addr] "Start address")
        (@arg log_filename: -l --log_file [file] "File to log executed instructions to")
        (@arg char_out_filename: --char_out_file [file] "File to write simulated character output to")
    )
    .get_matches()
}

fn process_u32_arg<'a>(
    args: &ArgMatches<'a>,
    name: &str,
    base: u32,
    default: u32,
) -> Result<u32, String> {
    let arg_str = match args.value_of(name) {
        Some(s) => s,
        None => return Ok(default),
    };

    u32::from_str_radix(arg_str, base).map_err(|e| format!("{} is malformed: {}", name, e))
}

fn process_arguments(args: &ArgMatches) -> Result<CliOpts, String> {
    let mut new_opts = CliOpts::default();

    new_opts.binary_file = args.value_of("binary_file").unwrap().to_string();
    new_opts.mem_size = process_u32_arg(args, "mem_size", 16, 0x100000)?;
    new_opts.mem_base = process_u32_arg(args, "mem_base", 16, 0x100000)?;
    new_opts.load_addr = process_u32_arg(args, "load_addr", 16, 0x100000)?;
    new_opts.start_addr = process_u32_arg(args, "start_addr", 16, new_opts.load_addr)?;
    new_opts.log_filename = args.value_of("log_filename").map(|s| s.to_string());
    new_opts.char_out_filename = args.value_of("char_out_filename").map(|s| s.to_string());

    Ok(new_opts)
}

struct SimEnvironment {
    memory_space: MemorySpace,
    hart_state: HartState,
    log_file: Option<Box<dyn Write>>,
    sim_ctrl_dev_idx: usize,
}

// Device that outputs characters written to it to stdout
struct CharOutputterDevice {
    char_out: Box<dyn Write>,
}

impl CharOutputterDevice {
    fn new(char_out_filename: Option<&str>) -> Result<Self, String> {
        Ok(CharOutputterDevice {
            char_out: match char_out_filename {
                Some(f) => Box::new(
                    File::create(f)
                        .map_err(|e| format!("Could not open char out file {}: {}", f, e))?,
                ),
                None => Box::new(io::stdout()),
            },
        })
    }
}

impl Memory for CharOutputterDevice {
    fn read_mem(&mut self, _addr: u32, _size: MemAccessSize) -> Option<u32> {
        Some(0x0)
    }

    fn write_mem(&mut self, _addr: u32, _size: MemAccessSize, store_data: u32) -> bool {
        let c: char = store_data as u8 as char;
        write!(self.char_out, "{}", c).expect("Failure writing character out");
        true
    }
}

// Device used to signal to the ISS to stop execution
struct SimulationCtrlDevice {
    stop: bool,
}

impl SimulationCtrlDevice {
    fn new() -> Self {
        SimulationCtrlDevice { stop: false }
    }
}

impl Memory for SimulationCtrlDevice {
    fn read_mem(&mut self, _addr: u32, _size: MemAccessSize) -> Option<u32> {
        Some(0x0)
    }

    fn write_mem(&mut self, _addr: u32, _size: MemAccessSize, store_data: u32) -> bool {
        if store_data != 0 {
            self.stop = true;
        } else {
            self.stop = false;
        }

        true
    }
}

fn load_binary(cli_opts: &CliOpts, mem: &mut impl Memory) -> Result<(), io::Error> {
    println!(
        "Loading binary {} to address {:08x}",
        cli_opts.binary_file, cli_opts.load_addr
    );

    let file = File::open(cli_opts.binary_file.as_str())?;
    let file_reader = BufReader::new(file);
    memories::read_to_memory(file_reader, mem, cli_opts.load_addr)?;

    Ok(())
}

fn setup_memory_space(cli_opts: &CliOpts) -> Result<MemorySpace, String> {
    let mut mem_space = MemorySpace::new();

    // TODO: Error handling
    mem_space
        .add_memory(
            cli_opts.mem_base,
            cli_opts.mem_size,
            Box::new(VecMemory::new(vec![0; cli_opts.mem_size as usize / 4])),
        )
        .expect("Adding base memory is expected to succeed");

    mem_space
        .add_memory(
            0x80000000,
            0x4,
            Box::new(CharOutputterDevice::new(
                cli_opts.char_out_filename.as_deref(),
            )?),
        )
        .expect("Adding char output device is expected to succeed");

    Ok(mem_space)
}

// Given CLI arguments sets up the simulation environment.
//
// Returned errors are strings describing the error.
fn setup_sim_environment(args: &ArgMatches) -> Result<SimEnvironment, String> {
    let cli_opts = process_arguments(args)?;

    let mut sim_environment = SimEnvironment {
        memory_space: setup_memory_space(&cli_opts)?,
        hart_state: HartState::new(),
        log_file: None,
        sim_ctrl_dev_idx: 0,
    };

    sim_environment.sim_ctrl_dev_idx = sim_environment
        .memory_space
        .add_memory(0x80000004, 0x4, Box::new(SimulationCtrlDevice::new()))
        .expect("Adding simulation control device is expected to succeed");

    sim_environment.hart_state.pc = cli_opts.start_addr;

    load_binary(&cli_opts, &mut sim_environment.memory_space)
        .map_err(|e| format!("Could not load binary {}: {}", cli_opts.binary_file, e))?;

    if let Some(log_filename) = cli_opts.log_filename {
        let log_file_unbuf = File::create(&log_filename)
            .map_err(|e| format!("Could not open log file {}: {}", log_filename, e))?;
        sim_environment.log_file = Some(Box::new(BufWriter::new(log_file_unbuf)));
    }

    Ok(sim_environment)
}

fn run_sim(sim_environment: &mut SimEnvironment) {
    let mut executor = InstructionExecutor {
        hart_state: &mut sim_environment.hart_state,
        mem: &mut sim_environment.memory_space,
    };

    let mut insn_count: u64 = 0;
    let start = Instant::now();

    loop {
        if let Some(log_file) = &mut sim_environment.log_file {
            // Output current instruction disassembly to log
            let insn_bits = executor
                .mem
                .read_mem(executor.hart_state.pc, MemAccessSize::Word)
                .unwrap_or_else(|| panic!("Could not read PC {:08x}", executor.hart_state.pc));

            let mut outputter = InstructionStringOutputter {
                insn_pc: executor.hart_state.pc,
            };

            writeln!(
                log_file,
                "{:x} {}",
                executor.hart_state.pc,
                rrs_lib::process_instruction(&mut outputter, insn_bits).unwrap()
            )
            .expect("Log file write failed");
        }

        // Execute instruction
        if let Err(trap) = executor.step() {
            if let Some(log_file) = &mut sim_environment.log_file {
                let log_line = match trap {
                    InstructionTrap::Exception(cause, val) => {
                        format!("[{:?} Exception, value:{:08x}]", cause, val)
                    }
                    InstructionTrap::Interrupt(irq) => format!("[Interrupt {:}]", irq),
                };

                writeln!(log_file, "{} @ PC {:08x}", log_line, executor.hart_state.pc)
                    .expect("Log file write failed");
            }
            executor.handle_trap(trap);
        }

        insn_count += 1;

        // Stop if stop requested by emulated binary via SimulationCtrlDevice
        if executor
            .mem
            .get_memory_ref::<SimulationCtrlDevice>(sim_environment.sim_ctrl_dev_idx)
            .unwrap()
            .stop
        {
            break;
        }

        if let Some(log_file) = &mut sim_environment.log_file {
            if let Some(reg_index) = executor.hart_state.last_register_write {
                // Output register written by instruction to log if it wrote to one
                writeln!(
                    log_file,
                    "\tx{} = {:08x}",
                    reg_index, executor.hart_state.registers[reg_index]
                )
                .expect("Log file write failed");
            }
        }
    }

    let elapsed = start.elapsed();
    let mhz = (insn_count as f64) / (elapsed.as_micros() as f64);
    println!(
        "{} instructions executed in {} ms {} MHz",
        insn_count,
        elapsed.as_millis(),
        mhz
    );
}

fn main() {
    let arg_matches = get_arg_matches();

    let mut sim_environment = match setup_sim_environment(&arg_matches) {
        Ok(se) => se,
        Err(e) => {
            println!("Failure starting simulation: {}", e);
            return;
        }
    };

    run_sim(&mut sim_environment);
}
