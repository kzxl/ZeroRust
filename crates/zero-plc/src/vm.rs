//! Deterministic Soft-PLC Virtual Machine runtime with zero dynamic allocation and hot-reload.

use crate::fb::{Ctu, PidCompact, Ton};
use crate::instruction::Instruction;
use crate::memory::PlcMemory;
use zero_core::error::{ZeroError, ZeroResult};

/// Registered standard Function Block wrapper for the VM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FunctionBlockInstance {
    /// Unallocated slot.
    Empty,
    /// TON On-Delay timer.
    Ton(Ton),
    /// CTU Up-counter.
    Ctu(Ctu),
    /// PID_Compact controller.
    Pid(PidCompact),
}

/// Statistics and diagnostics captured for each PLC execution scan cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlcCycleStats {
    /// Number of bytecode instructions executed in this cycle.
    pub instructions_executed: usize,
    /// Peak data stack depth reached during calculation.
    pub peak_stack_depth: usize,
    /// Execution status.
    pub halted_normally: bool,
}

/// Soft-PLC Bytecode Virtual Machine.
#[derive(Debug)]
pub struct PlcVm<
    const MAX_INSTR: usize,
    const MAX_FBS: usize,
    const IN_BYTES: usize,
    const OUT_BYTES: usize,
    const MARKER_BYTES: usize,
> {
    /// Bytecode instruction memory.
    pub program: [Instruction; MAX_INSTR],
    /// Length of currently loaded program.
    pub program_len: usize,
    /// Function block instances table.
    pub fbs: [FunctionBlockInstance; MAX_FBS],
    /// Process image and marker memory.
    pub memory: PlcMemory<IN_BYTES, OUT_BYTES, MARKER_BYTES>,
    /// Boolean accumulator: Current Result (CR) in IEC 61131-3 IL.
    pub cr: bool,
    /// Evaluation data stack for float arithmetic.
    stack: [f32; 32],
    /// Stack pointer.
    sp: usize,
}

impl<
        const MAX_INSTR: usize,
        const MAX_FBS: usize,
        const IN_BYTES: usize,
        const OUT_BYTES: usize,
        const MARKER_BYTES: usize,
    > Default for PlcVm<MAX_INSTR, MAX_FBS, IN_BYTES, OUT_BYTES, MARKER_BYTES>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
        const MAX_INSTR: usize,
        const MAX_FBS: usize,
        const IN_BYTES: usize,
        const OUT_BYTES: usize,
        const MARKER_BYTES: usize,
    > PlcVm<MAX_INSTR, MAX_FBS, IN_BYTES, OUT_BYTES, MARKER_BYTES>
{
    /// Creates a new PLC VM instance.
    pub fn new() -> Self {
        Self {
            program: [Instruction::Nop; MAX_INSTR],
            program_len: 0,
            fbs: [FunctionBlockInstance::Empty; MAX_FBS],
            memory: PlcMemory::new(),
            cr: false,
            stack: [0.0; 32],
            sp: 0,
        }
    }

    /// Loads a new program into the VM, resetting memory and accumulator state.
    pub fn load_program(&mut self, code: &[Instruction]) -> ZeroResult<()> {
        if code.len() > MAX_INSTR {
            return Err(ZeroError::BufferOverflow);
        }
        self.program[..code.len()].copy_from_slice(code);
        self.program_len = code.len();
        self.cr = false;
        self.sp = 0;
        Ok(())
    }

    /// Zero-downtime hot-reload: swaps the bytecode table without clearing memory or halting states.
    pub fn hot_reload_program(&mut self, code: &[Instruction]) -> ZeroResult<()> {
        if code.len() > MAX_INSTR {
            return Err(ZeroError::BufferOverflow);
        }
        self.program[..code.len()].copy_from_slice(code);
        self.program_len = code.len();
        Ok(())
    }

    /// Registers a Function Block instance into table slot `fb_id`.
    pub fn register_fb(&mut self, fb_id: u8, instance: FunctionBlockInstance) -> ZeroResult<()> {
        let idx = fb_id as usize;
        if idx >= MAX_FBS {
            return Err(ZeroError::InvalidArgument);
        }
        self.fbs[idx] = instance;
        Ok(())
    }

    fn push(&mut self, val: f32) -> ZeroResult<()> {
        if self.sp >= self.stack.len() {
            return Err(ZeroError::BufferFull);
        }
        self.stack[self.sp] = val;
        self.sp += 1;
        Ok(())
    }

    fn pop(&mut self) -> ZeroResult<f32> {
        if self.sp == 0 {
            return Err(ZeroError::BufferEmpty);
        }
        self.sp -= 1;
        Ok(self.stack[self.sp])
    }

    /// Executes one complete deterministic scan cycle.
    ///
    /// # Arguments
    /// - `dt_ms`: Scan cycle elapsed time in milliseconds.
    pub fn run_cycle(&mut self, dt_ms: u32) -> ZeroResult<PlcCycleStats> {
        let mut pc = 0;
        let mut instructions_executed = 0;
        let mut peak_stack_depth = self.sp;
        let mut halted = false;

        while pc < self.program_len {
            if self.sp > peak_stack_depth {
                peak_stack_depth = self.sp;
            }

            let instr = self.program[pc];
            instructions_executed += 1;

            match instr {
                Instruction::Nop => {
                    pc += 1;
                }
                Instruction::Ld(addr) => {
                    self.cr = self.memory.read_bit(addr)?;
                    pc += 1;
                }
                Instruction::LdN(addr) => {
                    self.cr = !self.memory.read_bit(addr)?;
                    pc += 1;
                }
                Instruction::St(addr) => {
                    self.memory.write_bit(addr, self.cr)?;
                    pc += 1;
                }
                Instruction::StN(addr) => {
                    self.memory.write_bit(addr, !self.cr)?;
                    pc += 1;
                }
                Instruction::S(addr) => {
                    if self.cr {
                        self.memory.write_bit(addr, true)?;
                    }
                    pc += 1;
                }
                Instruction::R(addr) => {
                    if self.cr {
                        self.memory.write_bit(addr, false)?;
                    }
                    pc += 1;
                }
                Instruction::And(addr) => {
                    let bit = self.memory.read_bit(addr)?;
                    self.cr = self.cr && bit;
                    pc += 1;
                }
                Instruction::AndN(addr) => {
                    let bit = self.memory.read_bit(addr)?;
                    self.cr = self.cr && !bit;
                    pc += 1;
                }
                Instruction::Or(addr) => {
                    let bit = self.memory.read_bit(addr)?;
                    self.cr = self.cr || bit;
                    pc += 1;
                }
                Instruction::OrN(addr) => {
                    let bit = self.memory.read_bit(addr)?;
                    self.cr = self.cr || !bit;
                    pc += 1;
                }
                Instruction::Xor(addr) => {
                    let bit = self.memory.read_bit(addr)?;
                    self.cr ^= bit;
                    pc += 1;
                }
                Instruction::XorN(addr) => {
                    let bit = self.memory.read_bit(addr)?;
                    self.cr ^= !bit;
                    pc += 1;
                }
                Instruction::LdVal(val) => {
                    self.push(val)?;
                    pc += 1;
                }
                Instruction::LdMem(addr) => {
                    let val = self.memory.read_f32(addr)?;
                    self.push(val)?;
                    pc += 1;
                }
                Instruction::StMem(addr) => {
                    let val = self.pop()?;
                    self.memory.write_f32(addr, val)?;
                    pc += 1;
                }
                Instruction::Add => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(a + b)?;
                    pc += 1;
                }
                Instruction::Sub => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(a - b)?;
                    pc += 1;
                }
                Instruction::Mul => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(a * b)?;
                    pc += 1;
                }
                Instruction::Div => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    if b.abs() < 1e-12 {
                        return Err(ZeroError::MathError);
                    }
                    self.push(a / b)?;
                    pc += 1;
                }
                Instruction::Eq => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.cr = (a - b).abs() < 1e-6;
                    pc += 1;
                }
                Instruction::Ne => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.cr = (a - b).abs() >= 1e-6;
                    pc += 1;
                }
                Instruction::Gt => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.cr = a > b;
                    pc += 1;
                }
                Instruction::Ge => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.cr = a >= b;
                    pc += 1;
                }
                Instruction::Lt => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.cr = a < b;
                    pc += 1;
                }
                Instruction::Le => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.cr = a <= b;
                    pc += 1;
                }
                Instruction::Jmp(target) => {
                    if target >= self.program_len {
                        return Err(ZeroError::InvalidArgument);
                    }
                    pc = target;
                }
                Instruction::JmpC(target) => {
                    if self.cr {
                        if target >= self.program_len {
                            return Err(ZeroError::InvalidArgument);
                        }
                        pc = target;
                    } else {
                        pc += 1;
                    }
                }
                Instruction::JmpCN(target) => {
                    if !self.cr {
                        if target >= self.program_len {
                            return Err(ZeroError::InvalidArgument);
                        }
                        pc = target;
                    } else {
                        pc += 1;
                    }
                }
                Instruction::CallFb(fb_id) => {
                    let idx = fb_id as usize;
                    if idx >= MAX_FBS {
                        return Err(ZeroError::InvalidArgument);
                    }
                    let mut fb = self.fbs[idx];
                    match &mut fb {
                        FunctionBlockInstance::Ton(ton) => {
                            self.cr = ton.update(self.cr, dt_ms);
                        }
                        FunctionBlockInstance::Ctu(ctu) => {
                            self.cr = ctu.update(self.cr, false);
                        }
                        FunctionBlockInstance::Pid(pid) => {
                            let input = self.pop().unwrap_or(0.0);
                            let sp = self.pop().unwrap_or(0.0);
                            let dt_sec = dt_ms as f32 / 1000.0;
                            let out = pid.update(sp, input, dt_sec);
                            self.push(out)?;
                        }
                        FunctionBlockInstance::Empty => {}
                    }
                    self.fbs[idx] = fb;
                    pc += 1;
                }
                Instruction::Halt => {
                    halted = true;
                    break;
                }
            }
        }

        Ok(PlcCycleStats {
            instructions_executed,
            peak_stack_depth,
            halted_normally: halted || pc >= self.program_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{PlcAddress, PlcArea};

    #[test]
    fn test_vm_start_stop_latch() {
        // Start pushbutton: %IX0.0
        // Stop pushbutton (NC): %IX0.1
        // Motor output: %QX0.0
        //
        // Logic:
        // LD %IX0.0
        // OR %QX0.0
        // AND %IX0.1
        // ST %QX0.0
        let start_btn = PlcAddress::bit(PlcArea::Inputs, 0, 0);
        let stop_btn = PlcAddress::bit(PlcArea::Inputs, 0, 1);
        let motor = PlcAddress::bit(PlcArea::Outputs, 0, 0);

        let program = [
            Instruction::Ld(start_btn),
            Instruction::Or(motor),
            Instruction::And(stop_btn),
            Instruction::St(motor),
            Instruction::Halt,
        ];

        let mut vm = PlcVm::<64, 4, 64, 64, 256>::new();
        vm.load_program(&program).unwrap();

        // 1. Initial: Stop button is closed (true), Start is open (false)
        vm.memory.write_bit(stop_btn, true).unwrap();
        vm.memory.write_bit(start_btn, false).unwrap();
        vm.run_cycle(10).unwrap();
        assert!(!vm.memory.read_bit(motor).unwrap());

        // 2. Press Start button
        vm.memory.write_bit(start_btn, true).unwrap();
        vm.run_cycle(10).unwrap();
        assert!(vm.memory.read_bit(motor).unwrap());

        // 3. Release Start button -> Motor remains sealed in (latched)
        vm.memory.write_bit(start_btn, false).unwrap();
        vm.run_cycle(10).unwrap();
        assert!(vm.memory.read_bit(motor).unwrap());

        // 4. Press Stop button (opens, false) -> Motor drops out
        vm.memory.write_bit(stop_btn, false).unwrap();
        vm.run_cycle(10).unwrap();
        assert!(!vm.memory.read_bit(motor).unwrap());
    }

    #[test]
    fn test_vm_hot_reload() {
        let flag = PlcAddress::bit(PlcArea::Markers, 0, 0);
        let out = PlcAddress::bit(PlcArea::Outputs, 0, 0);

        // Program 1: Set marker true
        let prog1 = [Instruction::LdVal(1.0), Instruction::Halt];
        let mut vm = PlcVm::<64, 4, 64, 64, 256>::new();
        vm.load_program(&prog1).unwrap();
        vm.memory.write_bit(flag, true).unwrap();

        // Program 2: Hot-reload, read flag to output
        let prog2 = [
            Instruction::Ld(flag),
            Instruction::St(out),
            Instruction::Halt,
        ];
        vm.hot_reload_program(&prog2).unwrap();

        // Marker state is preserved across hot reload
        vm.run_cycle(10).unwrap();
        assert!(vm.memory.read_bit(out).unwrap());
    }
}
