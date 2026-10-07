//! IEC 61131-3 Instruction List (IL) / Structured Text (ST) bytecode instruction set.

use crate::memory::PlcAddress;

/// Bytecode instructions executable by the deterministic Soft-PLC Virtual Machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    /// No operation.
    Nop,
    /// Boolean Load: Current Result (CR) = Memory[addr].
    Ld(PlcAddress),
    /// Boolean Load Negated: CR = !Memory[addr].
    LdN(PlcAddress),
    /// Boolean Store: Memory[addr] = CR.
    St(PlcAddress),
    /// Boolean Store Negated: Memory[addr] = !CR.
    StN(PlcAddress),
    /// Set / Latch: If CR is true, Memory[addr] = true.
    S(PlcAddress),
    /// Reset / Unlatch: If CR is true, Memory[addr] = false.
    R(PlcAddress),
    /// Boolean AND: CR = CR && Memory[addr].
    And(PlcAddress),
    /// Boolean AND Negated: CR = CR && !Memory[addr].
    AndN(PlcAddress),
    /// Boolean OR: CR = CR || Memory[addr].
    Or(PlcAddress),
    /// Boolean OR Negated: CR = CR || !Memory[addr].
    OrN(PlcAddress),
    /// Boolean XOR: CR = CR ^ Memory[addr].
    Xor(PlcAddress),
    /// Boolean XOR Negated: CR = CR ^ !Memory[addr].
    XorN(PlcAddress),

    /// Numeric Push Constant: Push immediate f32 onto data stack.
    LdVal(f32),
    /// Numeric Load Memory: Read 32-bit float from Memory[addr] and push to data stack.
    LdMem(PlcAddress),
    /// Numeric Store Memory: Pop f32 from data stack and write to Memory[addr].
    StMem(PlcAddress),

    /// Arithmetic Add: Pop B, pop A, push A + B.
    Add,
    /// Arithmetic Subtract: Pop B, pop A, push A - B.
    Sub,
    /// Arithmetic Multiply: Pop B, pop A, push A * B.
    Mul,
    /// Arithmetic Divide: Pop B, pop A, push A / B.
    Div,

    /// Comparison Equal: Pop B, pop A, CR = (A == B).
    Eq,
    /// Comparison Not Equal: Pop B, pop A, CR = (A != B).
    Ne,
    /// Comparison Greater Than: Pop B, pop A, CR = (A > B).
    Gt,
    /// Comparison Greater Than or Equal: Pop B, pop A, CR = (A >= B).
    Ge,
    /// Comparison Less Than: Pop B, pop A, CR = (A < B).
    Lt,
    /// Comparison Less Than or Equal: Pop B, pop A, CR = (A <= B).
    Le,

    /// Unconditional Jump to instruction index.
    Jmp(usize),
    /// Jump to instruction index if Current Result (CR) is True.
    JmpC(usize),
    /// Jump to instruction index if Current Result (CR) is False.
    JmpCN(usize),

    /// Call registered Function Block instance by ID.
    CallFb(u8),
    /// End of current program execution cycle.
    Halt,
}
