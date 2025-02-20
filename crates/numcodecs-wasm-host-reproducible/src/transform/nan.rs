use std::collections::VecDeque;

use anyhow::{anyhow, Error};
use wasm_encoder::reencode::{self, Reencode};

/// Adapted from cranelift's NaN canonicalisation codegen pass
/// <https://github.com/bytecodealliance/wasmtime/blob/ead6c7cc5dbb876437acbdf429a9190f25b96755/cranelift/codegen/src/nan_canonicalization.rs>
/// Released under the Apache-2.0 WITH LLVM-exception License
///
/// Implementation written referencing:
/// - WebAssembly Core Specification v2 <https://www.w3.org/TR/2024/WD-wasm-core-2-20240219>
/// - The "WebAssembly 128-bit packed SIMD Extension" Specification: <https://github.com/WebAssembly/spec/blob/f8114686035f6ffc358771c822ede3c96bf54cd9/proposals/simd/SIMD.md>
/// - The "Non-trapping Float-to-int Conversions" Extension Specification: <https://github.com/WebAssembly/spec/blob/f8114686035f6ffc358771c822ede3c96bf54cd9/proposals/nontrapping-float-to-int-conversion/Overview.md>
pub enum NaNCanonicaliser {}

impl NaNCanonicaliser {
    pub fn apply_to_module(
        wasm: &[u8],
        features: wasmparser::WasmFeatures,
    ) -> Result<Vec<u8>, anyhow::Error> {
        let mut parser = wasmparser::Parser::new(0);
        parser.set_features(features);

        let mut module = wasm_encoder::Module::new();

        let mut reencoder = NaNCanonicaliserReencoder {
            func_type_num_params: Vec::new(),
            function_type_indices: VecDeque::new(),
        };
        reencoder
            .parse_core_module(&mut module, parser, wasm)
            .map_err(|err| anyhow::format_err!("{}", err))?;

        let wasm = module.finish();
        wasmparser::Validator::new_with_features(features).validate_all(&wasm)?;
        Ok(wasm)
    }
}

struct NaNCanonicaliserReencoder {
    func_type_num_params: Vec<u32>,
    function_type_indices: VecDeque<usize>,
}

impl wasm_encoder::reencode::Reencode for NaNCanonicaliserReencoder {
    type Error = Error;

    fn parse_type_section(
        &mut self,
        types: &mut wasm_encoder::TypeSection,
        section: wasmparser::TypeSectionReader<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        for func_ty in section.into_iter_err_on_gc_types() {
            let func_ty = self.func_type(func_ty?)?;
            #[expect(clippy::cast_possible_truncation)] // guaranteed by wasm spec
            let num_params = func_ty.params().len() as u32;
            self.func_type_num_params.push(num_params);
            types.ty().func_type(&func_ty);
        }
        Ok(())
    }

    fn parse_function_section(
        &mut self,
        functions: &mut wasm_encoder::FunctionSection,
        section: wasmparser::FunctionSectionReader<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        for function in section {
            let function = self.type_index(function?);
            self.function_type_indices.push_back(function as usize);
            functions.function(function);
        }
        Ok(())
    }

    fn parse_function_body(
        &mut self,
        code: &mut wasm_encoder::CodeSection,
        func: wasmparser::FunctionBody<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        let Some(function_ty) = self.function_type_indices.pop_front() else {
            return Err(reencode::Error::UserError(anyhow!(
                "wasm function body without declaration",
            )));
        };
        let Some(num_params) = self.func_type_num_params.get(function_ty).copied() else {
            return Err(reencode::Error::UserError(anyhow!(
                "invalid type index for wasm function",
            )));
        };

        let locals = func.get_locals_reader()?;
        let locals = locals.into_iter().collect::<Result<Vec<_>, _>>()?;
        let mut locals = locals
            .into_iter()
            .map(|(count, ty)| self.val_type(ty).map(|ty| (count, ty)))
            .collect::<Result<Vec<_>, _>>()?;
        let mut num_locals = locals.iter().map(|(count, _ty)| *count).sum::<u32>();

        let mut stash_f32 = None;
        let mut stash_f64 = None;
        let mut stash_v128 = None;

        let instructions = func.get_operators_reader()?;
        for instruction in instructions {
            if let Some(kind) = Self::may_produce_non_deterministic_nan(&instruction?)
                .map_err(|err| reencode::Error::UserError(anyhow!(err)))?
            {
                match kind {
                    MaybeNaNKind::F32 => stash_f32.get_or_insert_with(|| {
                        let stash_f32 = num_params + num_locals;
                        locals.push((1, wasm_encoder::ValType::F32));
                        num_locals += 1;
                        stash_f32
                    }),
                    MaybeNaNKind::F64 => stash_f64.get_or_insert_with(|| {
                        let stash_f64 = num_params + num_locals;
                        locals.push((1, wasm_encoder::ValType::F64));
                        num_locals += 1;
                        stash_f64
                    }),
                    MaybeNaNKind::F32x4 | MaybeNaNKind::F64x2 => {
                        stash_v128.get_or_insert_with(|| {
                            let stash_v128 = num_params + num_locals;
                            locals.push((1, wasm_encoder::ValType::V128));
                            num_locals += 1;
                            stash_v128
                        })
                    }
                };
            }
        }

        let mut function = wasm_encoder::Function::new(locals);

        for instruction in func.get_operators_reader()? {
            let instruction = instruction?;

            let kind = Self::may_produce_non_deterministic_nan(&instruction)
                .map_err(|err| reencode::Error::UserError(anyhow!(err)))?;

            function.instruction(&self.instruction(instruction)?);

            if let Some(kind) = kind {
                let Some(stash) = (match kind {
                    MaybeNaNKind::F32 => stash_f32,
                    MaybeNaNKind::F64 => stash_f64,
                    MaybeNaNKind::F32x4 | MaybeNaNKind::F64x2 => stash_v128,
                }) else {
                    return Err(reencode::Error::UserError(anyhow!(
                        "wasm float operation without matching canonicalisation stash",
                    )));
                };

                for canon_instruction in
                    Self::generate_nan_canonicalisation_instructions(kind, stash)
                {
                    function.instruction(&canon_instruction);
                }
            }
        }

        code.function(&function);

        Ok(())
    }
}

impl NaNCanonicaliserReencoder {
    // Canonical 32-bit and 64-bit NaN values
    const CANON_NAN_B32: u32 = 0x7FC0_0000;
    const CANON_NAN_B32X4: u128 = 0x7FC0_0000_7FC0_0000_7FC0_0000_7FC0_0000;
    const CANON_NAN_B64: u64 = 0x7FF8_0000_0000_0000;
    const CANON_NAN_B64X2: u128 = 0x7FF8_0000_0000_0000_7FF8_0000_0000_0000;

    fn generate_nan_canonicalisation_instructions(
        kind: MaybeNaNKind,
        stash: u32,
    ) -> impl IntoIterator<Item = wasm_encoder::Instruction<'static>> {
        match kind {
            MaybeNaNKind::F32 => [
                // stack: [x, ...]; stash: ??
                wasm_encoder::Instruction::LocalSet(stash),
                // stack: [...]; stash: x
                // canonical NaN
                wasm_encoder::Instruction::F32Const(f32::from_bits(Self::CANON_NAN_B32)),
                // stack: [canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, x, canon(NaN), ...]; stash: x
                // isNaN: x != x
                wasm_encoder::Instruction::F32Ne,
                // stack: [isNaN, x, canon(NaN), ...]; stash: x
                // select expects the stack [c: isNaN, val2: x, val1: canon(NaN), ...]
                // select returns if c == 0 { val2 } else { val1 }
                // here if isNaN  then c = 1 and val1 = canon(NaN) is returned
                //      if !isNaN then c = 0 and val2 = x is returned
                wasm_encoder::Instruction::Select,
                // stack: [canon(x), ...]; stash: x
            ],
            MaybeNaNKind::F64 => [
                // stack: [x, ...]; stash: ??
                wasm_encoder::Instruction::LocalSet(stash),
                // stack: [...]; stash: x
                // canonical NaN
                wasm_encoder::Instruction::F64Const(f64::from_bits(Self::CANON_NAN_B64)),
                // stack: [canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, x, canon(NaN), ...]; stash: x
                // isNaN: x != x
                wasm_encoder::Instruction::F64Ne,
                // stack: [isNaN, x, canon(NaN), ...]; stash: x
                // select expects the stack [c: isNaN, val2: x, val1: canon(NaN), ...]
                // select returns if c == 0 { val2 } else { val1 }
                // here if isNaN  then c = 1 and val1 = canon(NaN) is returned
                //      if !isNaN then c = 0 and val2 = x is returned
                wasm_encoder::Instruction::Select,
                // stack: [canon(x), ...]; stash: x
            ],
            MaybeNaNKind::F32x4 => [
                // stack: [x, ...]; stash: ??
                wasm_encoder::Instruction::LocalSet(stash),
                // stack: [...]; stash: x
                // canonical NaN
                wasm_encoder::Instruction::V128Const(i128::from_le_bytes(
                    Self::CANON_NAN_B32X4.to_le_bytes(),
                )),
                // stack: [canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, x, canon(NaN), ...]; stash: x
                // isNaN: x != x
                wasm_encoder::Instruction::F32x4Ne,
                // stack: [isNaN, x, canon(NaN), ...]; stash: x
                // bitselect expects the stack [c: isNaN, val2: x, val1: canon(NaN), ...]
                // bitselect returns if c[i] == 0 { val2[i] } else { val1[i] }
                // here if per-lane isNaN  then c = 1 and val1 = canon(NaN) is returned
                //      if per-lane !isNaN then c = 0 and val2 = x is returned
                wasm_encoder::Instruction::V128Bitselect,
                // stack: [canon(x), ...]; stash: x
            ],
            MaybeNaNKind::F64x2 => [
                // stack: [x, ...]; stash: ??
                wasm_encoder::Instruction::LocalSet(stash),
                // stack: [...]; stash: x
                // canonical NaN
                wasm_encoder::Instruction::V128Const(i128::from_le_bytes(
                    Self::CANON_NAN_B64X2.to_le_bytes(),
                )),
                // stack: [canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, canon(NaN), ...]; stash: x
                wasm_encoder::Instruction::LocalGet(stash),
                // stack: [x, x, x, canon(NaN), ...]; stash: x
                // isNaN: x != x
                wasm_encoder::Instruction::F64x2Ne,
                // stack: [isNaN, x, canon(NaN), ...]; stash: x
                // bitselect expects the stack [c: isNaN, val2: x, val1: canon(NaN), ...]
                // bitselect returns if c[i] == 0 { val2[i] } else { val1[i] }
                // here if per-lane isNaN  then c = 1 and val1 = canon(NaN) is returned
                //      if per-lane !isNaN then c = 0 and val2 = x is returned
                wasm_encoder::Instruction::V128Bitselect,
                // stack: [canon(x), ...]; stash: x
            ],
        }
    }

    #[expect(clippy::too_many_lines)]
    fn may_produce_non_deterministic_nan(
        instr: &wasmparser::Operator,
    ) -> Result<Option<MaybeNaNKind>, NonDeterministicWasmFeature> {
        match instr {
            // === MVP ===
            // non-float operation
            wasmparser::Operator::Unreachable
            | wasmparser::Operator::Nop
            | wasmparser::Operator::Block { .. }
            | wasmparser::Operator::Loop { .. }
            | wasmparser::Operator::If { .. }
            | wasmparser::Operator::Else => Ok(None),
            // === Exception handling ===
            // non-float operation
            wasmparser::Operator::TryTable { .. }
            | wasmparser::Operator::Throw { .. }
            | wasmparser::Operator::ThrowRef => Ok(None),
            // === Legacy exception handling (deprecated) ===
            // non-float operation
            wasmparser::Operator::Try { .. }
            | wasmparser::Operator::Catch { .. }
            | wasmparser::Operator::Rethrow { .. }
            | wasmparser::Operator::Delegate { .. }
            | wasmparser::Operator::CatchAll => Ok(None),
            // === MVP ===
            // non-float operation
            wasmparser::Operator::End
            | wasmparser::Operator::Br { .. }
            | wasmparser::Operator::BrIf { .. }
            | wasmparser::Operator::BrTable { .. }
            | wasmparser::Operator::Return
            | wasmparser::Operator::Call { .. }
            | wasmparser::Operator::CallIndirect { .. } => Ok(None),
            // === Tail calls ===
            // non-float operation
            wasmparser::Operator::ReturnCall { .. }
            | wasmparser::Operator::ReturnCallIndirect { .. } => Ok(None),
            // === MVP ===
            // non-float operation
            wasmparser::Operator::Drop | wasmparser::Operator::Select => Ok(None),
            // === Reference types ===
            // non-float operation
            wasmparser::Operator::TypedSelect { .. } => Ok(None),
            // locals may contain floats, but get/set/tee are deterministic
            wasmparser::Operator::LocalGet { .. }
            | wasmparser::Operator::LocalSet { .. }
            | wasmparser::Operator::LocalTee { .. } => Ok(None),
            // non-float operation
            wasmparser::Operator::GlobalGet { .. }
            | wasmparser::Operator::GlobalSet { .. }
            | wasmparser::Operator::I32Load { .. }
            | wasmparser::Operator::I64Load { .. } => Ok(None),
            // loading a float from memory is deterministic, even when a
            //  non-canonical NaN value is loaded
            wasmparser::Operator::F32Load { .. } | wasmparser::Operator::F64Load { .. } => Ok(None),
            // non-float operation
            wasmparser::Operator::I32Load8S { .. }
            | wasmparser::Operator::I32Load8U { .. }
            | wasmparser::Operator::I32Load16S { .. }
            | wasmparser::Operator::I32Load16U { .. }
            | wasmparser::Operator::I64Load8S { .. }
            | wasmparser::Operator::I64Load8U { .. }
            | wasmparser::Operator::I64Load16S { .. }
            | wasmparser::Operator::I64Load16U { .. }
            | wasmparser::Operator::I64Load32S { .. }
            | wasmparser::Operator::I64Load32U { .. }
            | wasmparser::Operator::I32Store { .. }
            | wasmparser::Operator::I64Store { .. } => Ok(None),
            // storing a float to memory is deterministic, even when a
            //  non-canonical NaN value is stored
            wasmparser::Operator::F32Store { .. } | wasmparser::Operator::F64Store { .. } => {
                Ok(None)
            }
            // non-float operation
            wasmparser::Operator::I32Store8 { .. }
            | wasmparser::Operator::I32Store16 { .. }
            | wasmparser::Operator::I64Store8 { .. }
            | wasmparser::Operator::I64Store16 { .. }
            | wasmparser::Operator::I64Store32 { .. }
            | wasmparser::Operator::MemorySize { .. }
            | wasmparser::Operator::MemoryGrow { .. }
            | wasmparser::Operator::I32Const { .. }
            | wasmparser::Operator::I64Const { .. } => Ok(None),
            // constant float values are deterministic, even when they represent
            //  non-canonical NaN values
            wasmparser::Operator::F32Const { .. } | wasmparser::Operator::F64Const { .. } => {
                Ok(None)
            }
            // === Reference types ===
            // non-float operation
            wasmparser::Operator::RefNull { .. }
            | wasmparser::Operator::RefIsNull
            | wasmparser::Operator::RefFunc { .. } => Ok(None),
            // === Garbage collection ===
            // non-float operation
            wasmparser::Operator::RefEq => Ok(None),
            // === MVP ===
            // non-float operation
            wasmparser::Operator::I32Eqz
            | wasmparser::Operator::I32Eq
            | wasmparser::Operator::I32Ne
            | wasmparser::Operator::I32LtS
            | wasmparser::Operator::I32LtU
            | wasmparser::Operator::I32GtS
            | wasmparser::Operator::I32GtU
            | wasmparser::Operator::I32LeS
            | wasmparser::Operator::I32LeU
            | wasmparser::Operator::I32GeS
            | wasmparser::Operator::I32GeU
            | wasmparser::Operator::I64Eqz
            | wasmparser::Operator::I64Eq
            | wasmparser::Operator::I64Ne
            | wasmparser::Operator::I64LtS
            | wasmparser::Operator::I64LtU
            | wasmparser::Operator::I64GtS
            | wasmparser::Operator::I64GtU
            | wasmparser::Operator::I64LeS
            | wasmparser::Operator::I64LeU
            | wasmparser::Operator::I64GeS
            | wasmparser::Operator::I64GeU => Ok(None),
            // deterministic float operation
            wasmparser::Operator::F32Eq
            | wasmparser::Operator::F32Ne
            | wasmparser::Operator::F32Lt
            | wasmparser::Operator::F32Gt
            | wasmparser::Operator::F32Le
            | wasmparser::Operator::F32Ge
            | wasmparser::Operator::F64Eq
            | wasmparser::Operator::F64Ne
            | wasmparser::Operator::F64Lt
            | wasmparser::Operator::F64Gt
            | wasmparser::Operator::F64Le
            | wasmparser::Operator::F64Ge => Ok(None),
            // non-float operation
            wasmparser::Operator::I32Clz
            | wasmparser::Operator::I32Ctz
            | wasmparser::Operator::I32Popcnt
            | wasmparser::Operator::I32Add
            | wasmparser::Operator::I32Sub
            | wasmparser::Operator::I32Mul
            | wasmparser::Operator::I32DivS
            | wasmparser::Operator::I32DivU
            | wasmparser::Operator::I32RemS
            | wasmparser::Operator::I32RemU
            | wasmparser::Operator::I32And
            | wasmparser::Operator::I32Or
            | wasmparser::Operator::I32Xor
            | wasmparser::Operator::I32Shl
            | wasmparser::Operator::I32ShrS
            | wasmparser::Operator::I32ShrU
            | wasmparser::Operator::I32Rotl
            | wasmparser::Operator::I32Rotr
            | wasmparser::Operator::I64Clz
            | wasmparser::Operator::I64Ctz
            | wasmparser::Operator::I64Popcnt
            | wasmparser::Operator::I64Add
            | wasmparser::Operator::I64Sub
            | wasmparser::Operator::I64Mul
            | wasmparser::Operator::I64DivS
            | wasmparser::Operator::I64DivU
            | wasmparser::Operator::I64RemS
            | wasmparser::Operator::I64RemU
            | wasmparser::Operator::I64And
            | wasmparser::Operator::I64Or
            | wasmparser::Operator::I64Xor
            | wasmparser::Operator::I64Shl
            | wasmparser::Operator::I64ShrS
            | wasmparser::Operator::I64ShrU
            | wasmparser::Operator::I64Rotl
            | wasmparser::Operator::I64Rotr => Ok(None),
            // deterministic float operation
            wasmparser::Operator::F32Abs | wasmparser::Operator::F32Neg => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F32Ceil
            | wasmparser::Operator::F32Floor
            | wasmparser::Operator::F32Trunc
            | wasmparser::Operator::F32Nearest
            | wasmparser::Operator::F32Sqrt
            | wasmparser::Operator::F32Add
            | wasmparser::Operator::F32Sub
            | wasmparser::Operator::F32Mul
            | wasmparser::Operator::F32Div
            | wasmparser::Operator::F32Min
            | wasmparser::Operator::F32Max => Ok(Some(MaybeNaNKind::F32)),
            // deterministic float operation
            wasmparser::Operator::F32Copysign
            | wasmparser::Operator::F64Abs
            | wasmparser::Operator::F64Neg => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F64Ceil
            | wasmparser::Operator::F64Floor
            | wasmparser::Operator::F64Trunc
            | wasmparser::Operator::F64Nearest
            | wasmparser::Operator::F64Sqrt
            | wasmparser::Operator::F64Add
            | wasmparser::Operator::F64Sub
            | wasmparser::Operator::F64Mul
            | wasmparser::Operator::F64Div
            | wasmparser::Operator::F64Min
            | wasmparser::Operator::F64Max => Ok(Some(MaybeNaNKind::F64)),
            // deterministic float operation
            wasmparser::Operator::F64Copysign => Ok(None),
            // non-float operation
            wasmparser::Operator::I32WrapI64 => Ok(None),
            // truncate float to int, which traps to be deterministic
            wasmparser::Operator::I32TruncF32S
            | wasmparser::Operator::I32TruncF32U
            | wasmparser::Operator::I32TruncF64S
            | wasmparser::Operator::I32TruncF64U => Ok(None),
            // non-float operation
            wasmparser::Operator::I64ExtendI32S | wasmparser::Operator::I64ExtendI32U => Ok(None),
            // truncate float to int, which traps to be deterministic
            wasmparser::Operator::I64TruncF32S
            | wasmparser::Operator::I64TruncF32U
            | wasmparser::Operator::I64TruncF64S
            | wasmparser::Operator::I64TruncF64U => Ok(None),
            // convert int to float, deterministic
            wasmparser::Operator::F32ConvertI32S
            | wasmparser::Operator::F32ConvertI32U
            | wasmparser::Operator::F32ConvertI64S
            | wasmparser::Operator::F32ConvertI64U => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F32DemoteF64 => Ok(Some(MaybeNaNKind::F32)),
            // convert int to float, deterministic
            wasmparser::Operator::F64ConvertI32S
            | wasmparser::Operator::F64ConvertI32U
            | wasmparser::Operator::F64ConvertI64S
            | wasmparser::Operator::F64ConvertI64U => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F64PromoteF32 => Ok(Some(MaybeNaNKind::F64)),
            // float <-> int bit cast, deterministic
            wasmparser::Operator::I32ReinterpretF32
            | wasmparser::Operator::I64ReinterpretF64
            | wasmparser::Operator::F32ReinterpretI32
            | wasmparser::Operator::F64ReinterpretI64 => Ok(None),
            // === Sign extension ===
            // non-float operation
            wasmparser::Operator::I32Extend8S
            | wasmparser::Operator::I32Extend16S
            | wasmparser::Operator::I64Extend8S
            | wasmparser::Operator::I64Extend16S
            | wasmparser::Operator::I64Extend32S => Ok(None),
            // === Garbage collection ===
            // non-float operation
            wasmparser::Operator::StructNew { .. }
            | wasmparser::Operator::StructNewDefault { .. }
            | wasmparser::Operator::StructGet { .. }
            | wasmparser::Operator::StructGetS { .. }
            | wasmparser::Operator::StructGetU { .. }
            | wasmparser::Operator::StructSet { .. }
            | wasmparser::Operator::ArrayNew { .. }
            | wasmparser::Operator::ArrayNewDefault { .. }
            | wasmparser::Operator::ArrayNewFixed { .. }
            | wasmparser::Operator::ArrayNewData { .. }
            | wasmparser::Operator::ArrayNewElem { .. }
            | wasmparser::Operator::ArrayGet { .. }
            | wasmparser::Operator::ArrayGetS { .. }
            | wasmparser::Operator::ArrayGetU { .. }
            | wasmparser::Operator::ArraySet { .. }
            | wasmparser::Operator::ArrayLen
            | wasmparser::Operator::ArrayFill { .. }
            | wasmparser::Operator::ArrayCopy { .. }
            | wasmparser::Operator::ArrayInitData { .. }
            | wasmparser::Operator::ArrayInitElem { .. }
            | wasmparser::Operator::RefTestNonNull { .. }
            | wasmparser::Operator::RefTestNullable { .. }
            | wasmparser::Operator::RefCastNonNull { .. }
            | wasmparser::Operator::RefCastNullable { .. }
            | wasmparser::Operator::BrOnCast { .. }
            | wasmparser::Operator::BrOnCastFail { .. }
            | wasmparser::Operator::AnyConvertExtern
            | wasmparser::Operator::ExternConvertAny
            | wasmparser::Operator::RefI31
            | wasmparser::Operator::I31GetS
            | wasmparser::Operator::I31GetU => Ok(None),
            // === Non-trapping float-to-int conversions ===
            // truncate float to int, which saturates to be deterministic
            wasmparser::Operator::I32TruncSatF32S
            | wasmparser::Operator::I32TruncSatF32U
            | wasmparser::Operator::I32TruncSatF64S
            | wasmparser::Operator::I32TruncSatF64U
            | wasmparser::Operator::I64TruncSatF32S
            | wasmparser::Operator::I64TruncSatF32U
            | wasmparser::Operator::I64TruncSatF64S
            | wasmparser::Operator::I64TruncSatF64U => Ok(None),
            // === Bulk memory ===
            // non-float operation
            wasmparser::Operator::MemoryInit { .. }
            | wasmparser::Operator::DataDrop { .. }
            | wasmparser::Operator::MemoryCopy { .. }
            | wasmparser::Operator::MemoryFill { .. }
            | wasmparser::Operator::TableInit { .. }
            | wasmparser::Operator::ElemDrop { .. }
            | wasmparser::Operator::TableCopy { .. } => Ok(None),
            // === Reference types ===
            // non-float operation
            wasmparser::Operator::TableFill { .. }
            | wasmparser::Operator::TableGet { .. }
            | wasmparser::Operator::TableSet { .. }
            | wasmparser::Operator::TableGrow { .. }
            | wasmparser::Operator::TableSize { .. } => Ok(None),
            // === Memory control ===
            // non-float operation
            wasmparser::Operator::MemoryDiscard { .. } => Ok(None),
            // === Threads ===
            // non-deterministic with potential data races
            wasmparser::Operator::MemoryAtomicNotify { .. }
            | wasmparser::Operator::MemoryAtomicWait32 { .. }
            | wasmparser::Operator::MemoryAtomicWait64 { .. }
            | wasmparser::Operator::AtomicFence
            | wasmparser::Operator::I32AtomicLoad { .. }
            | wasmparser::Operator::I64AtomicLoad { .. }
            | wasmparser::Operator::I32AtomicLoad8U { .. }
            | wasmparser::Operator::I32AtomicLoad16U { .. }
            | wasmparser::Operator::I64AtomicLoad8U { .. }
            | wasmparser::Operator::I64AtomicLoad16U { .. }
            | wasmparser::Operator::I64AtomicLoad32U { .. }
            | wasmparser::Operator::I32AtomicStore { .. }
            | wasmparser::Operator::I64AtomicStore { .. }
            | wasmparser::Operator::I32AtomicStore8 { .. }
            | wasmparser::Operator::I32AtomicStore16 { .. }
            | wasmparser::Operator::I64AtomicStore8 { .. }
            | wasmparser::Operator::I64AtomicStore16 { .. }
            | wasmparser::Operator::I64AtomicStore32 { .. }
            | wasmparser::Operator::I32AtomicRmwAdd { .. }
            | wasmparser::Operator::I64AtomicRmwAdd { .. }
            | wasmparser::Operator::I32AtomicRmw8AddU { .. }
            | wasmparser::Operator::I32AtomicRmw16AddU { .. }
            | wasmparser::Operator::I64AtomicRmw8AddU { .. }
            | wasmparser::Operator::I64AtomicRmw16AddU { .. }
            | wasmparser::Operator::I64AtomicRmw32AddU { .. }
            | wasmparser::Operator::I32AtomicRmwSub { .. }
            | wasmparser::Operator::I64AtomicRmwSub { .. }
            | wasmparser::Operator::I32AtomicRmw8SubU { .. }
            | wasmparser::Operator::I32AtomicRmw16SubU { .. }
            | wasmparser::Operator::I64AtomicRmw8SubU { .. }
            | wasmparser::Operator::I64AtomicRmw16SubU { .. }
            | wasmparser::Operator::I64AtomicRmw32SubU { .. }
            | wasmparser::Operator::I32AtomicRmwAnd { .. }
            | wasmparser::Operator::I64AtomicRmwAnd { .. }
            | wasmparser::Operator::I32AtomicRmw8AndU { .. }
            | wasmparser::Operator::I32AtomicRmw16AndU { .. }
            | wasmparser::Operator::I64AtomicRmw8AndU { .. }
            | wasmparser::Operator::I64AtomicRmw16AndU { .. }
            | wasmparser::Operator::I64AtomicRmw32AndU { .. }
            | wasmparser::Operator::I32AtomicRmwOr { .. }
            | wasmparser::Operator::I64AtomicRmwOr { .. }
            | wasmparser::Operator::I32AtomicRmw8OrU { .. }
            | wasmparser::Operator::I32AtomicRmw16OrU { .. }
            | wasmparser::Operator::I64AtomicRmw8OrU { .. }
            | wasmparser::Operator::I64AtomicRmw16OrU { .. }
            | wasmparser::Operator::I64AtomicRmw32OrU { .. }
            | wasmparser::Operator::I32AtomicRmwXor { .. }
            | wasmparser::Operator::I64AtomicRmwXor { .. }
            | wasmparser::Operator::I32AtomicRmw8XorU { .. }
            | wasmparser::Operator::I32AtomicRmw16XorU { .. }
            | wasmparser::Operator::I64AtomicRmw8XorU { .. }
            | wasmparser::Operator::I64AtomicRmw16XorU { .. }
            | wasmparser::Operator::I64AtomicRmw32XorU { .. }
            | wasmparser::Operator::I32AtomicRmwXchg { .. }
            | wasmparser::Operator::I64AtomicRmwXchg { .. }
            | wasmparser::Operator::I32AtomicRmw8XchgU { .. }
            | wasmparser::Operator::I32AtomicRmw16XchgU { .. }
            | wasmparser::Operator::I64AtomicRmw8XchgU { .. }
            | wasmparser::Operator::I64AtomicRmw16XchgU { .. }
            | wasmparser::Operator::I64AtomicRmw32XchgU { .. }
            | wasmparser::Operator::I32AtomicRmwCmpxchg { .. }
            | wasmparser::Operator::I64AtomicRmwCmpxchg { .. }
            | wasmparser::Operator::I32AtomicRmw8CmpxchgU { .. }
            | wasmparser::Operator::I32AtomicRmw16CmpxchgU { .. }
            | wasmparser::Operator::I64AtomicRmw8CmpxchgU { .. }
            | wasmparser::Operator::I64AtomicRmw16CmpxchgU { .. }
            | wasmparser::Operator::I64AtomicRmw32CmpxchgU { .. } => {
                Err(NonDeterministicWasmFeature::Threads)
            }
            // === Shared-everything threads ===
            // non-deterministic with potential data races
            wasmparser::Operator::GlobalAtomicGet { .. }
            | wasmparser::Operator::GlobalAtomicSet { .. }
            | wasmparser::Operator::GlobalAtomicRmwAdd { .. }
            | wasmparser::Operator::GlobalAtomicRmwSub { .. }
            | wasmparser::Operator::GlobalAtomicRmwAnd { .. }
            | wasmparser::Operator::GlobalAtomicRmwOr { .. }
            | wasmparser::Operator::GlobalAtomicRmwXor { .. }
            | wasmparser::Operator::GlobalAtomicRmwXchg { .. }
            | wasmparser::Operator::GlobalAtomicRmwCmpxchg { .. }
            | wasmparser::Operator::TableAtomicGet { .. }
            | wasmparser::Operator::TableAtomicSet { .. }
            | wasmparser::Operator::TableAtomicRmwXchg { .. }
            | wasmparser::Operator::TableAtomicRmwCmpxchg { .. }
            | wasmparser::Operator::StructAtomicGet { .. }
            | wasmparser::Operator::StructAtomicGetS { .. }
            | wasmparser::Operator::StructAtomicGetU { .. }
            | wasmparser::Operator::StructAtomicSet { .. }
            | wasmparser::Operator::StructAtomicRmwAdd { .. }
            | wasmparser::Operator::StructAtomicRmwSub { .. }
            | wasmparser::Operator::StructAtomicRmwAnd { .. }
            | wasmparser::Operator::StructAtomicRmwOr { .. }
            | wasmparser::Operator::StructAtomicRmwXor { .. }
            | wasmparser::Operator::StructAtomicRmwXchg { .. }
            | wasmparser::Operator::StructAtomicRmwCmpxchg { .. }
            | wasmparser::Operator::ArrayAtomicGet { .. }
            | wasmparser::Operator::ArrayAtomicGetS { .. }
            | wasmparser::Operator::ArrayAtomicGetU { .. }
            | wasmparser::Operator::ArrayAtomicSet { .. }
            | wasmparser::Operator::ArrayAtomicRmwAdd { .. }
            | wasmparser::Operator::ArrayAtomicRmwSub { .. }
            | wasmparser::Operator::ArrayAtomicRmwAnd { .. }
            | wasmparser::Operator::ArrayAtomicRmwOr { .. }
            | wasmparser::Operator::ArrayAtomicRmwXor { .. }
            | wasmparser::Operator::ArrayAtomicRmwXchg { .. }
            | wasmparser::Operator::ArrayAtomicRmwCmpxchg { .. }
            | wasmparser::Operator::RefI31Shared => {
                Err(NonDeterministicWasmFeature::SharedEverythingThreads)
            }
            // === SIMD ===
            // non-float operation, memory loads/stores are deterministic
            wasmparser::Operator::V128Load { .. }
            | wasmparser::Operator::V128Load8x8S { .. }
            | wasmparser::Operator::V128Load8x8U { .. }
            | wasmparser::Operator::V128Load16x4S { .. }
            | wasmparser::Operator::V128Load16x4U { .. }
            | wasmparser::Operator::V128Load32x2S { .. }
            | wasmparser::Operator::V128Load32x2U { .. }
            | wasmparser::Operator::V128Load8Splat { .. }
            | wasmparser::Operator::V128Load16Splat { .. }
            | wasmparser::Operator::V128Load32Splat { .. }
            | wasmparser::Operator::V128Load64Splat { .. }
            | wasmparser::Operator::V128Load32Zero { .. }
            | wasmparser::Operator::V128Load64Zero { .. }
            | wasmparser::Operator::V128Store { .. }
            | wasmparser::Operator::V128Load8Lane { .. }
            | wasmparser::Operator::V128Load16Lane { .. }
            | wasmparser::Operator::V128Load32Lane { .. }
            | wasmparser::Operator::V128Load64Lane { .. }
            | wasmparser::Operator::V128Store8Lane { .. }
            | wasmparser::Operator::V128Store16Lane { .. }
            | wasmparser::Operator::V128Store32Lane { .. }
            | wasmparser::Operator::V128Store64Lane { .. } => Ok(None),
            // constant values are deterministic, even when they represent
            //  non-canonical NaN values
            wasmparser::Operator::V128Const { .. } => Ok(None),
            // non-float operation
            wasmparser::Operator::I8x16Shuffle { .. }
            | wasmparser::Operator::I8x16ExtractLaneS { .. }
            | wasmparser::Operator::I8x16ExtractLaneU { .. }
            | wasmparser::Operator::I8x16ReplaceLane { .. }
            | wasmparser::Operator::I16x8ExtractLaneS { .. }
            | wasmparser::Operator::I16x8ExtractLaneU { .. }
            | wasmparser::Operator::I16x8ReplaceLane { .. }
            | wasmparser::Operator::I32x4ExtractLane { .. }
            | wasmparser::Operator::I32x4ReplaceLane { .. }
            | wasmparser::Operator::I64x2ExtractLane { .. }
            | wasmparser::Operator::I64x2ReplaceLane { .. } => Ok(None),
            // extracting or replacing lanes is deterministic
            wasmparser::Operator::F32x4ExtractLane { .. }
            | wasmparser::Operator::F32x4ReplaceLane { .. }
            | wasmparser::Operator::F64x2ExtractLane { .. }
            | wasmparser::Operator::F64x2ReplaceLane { .. } => Ok(None),
            // non-float operation
            wasmparser::Operator::I8x16Swizzle
            | wasmparser::Operator::I8x16Splat
            | wasmparser::Operator::I16x8Splat
            | wasmparser::Operator::I32x4Splat
            | wasmparser::Operator::I64x2Splat => Ok(None),
            // splatting is deterministic, even when a non-canonical NaN value
            //  is splatted
            wasmparser::Operator::F32x4Splat | wasmparser::Operator::F64x2Splat => Ok(None),
            // non-float operation
            wasmparser::Operator::I8x16Eq
            | wasmparser::Operator::I8x16Ne
            | wasmparser::Operator::I8x16LtS
            | wasmparser::Operator::I8x16LtU
            | wasmparser::Operator::I8x16GtS
            | wasmparser::Operator::I8x16GtU
            | wasmparser::Operator::I8x16LeS
            | wasmparser::Operator::I8x16LeU
            | wasmparser::Operator::I8x16GeS
            | wasmparser::Operator::I8x16GeU
            | wasmparser::Operator::I16x8Eq
            | wasmparser::Operator::I16x8Ne
            | wasmparser::Operator::I16x8LtS
            | wasmparser::Operator::I16x8LtU
            | wasmparser::Operator::I16x8GtS
            | wasmparser::Operator::I16x8GtU
            | wasmparser::Operator::I16x8LeS
            | wasmparser::Operator::I16x8LeU
            | wasmparser::Operator::I16x8GeS
            | wasmparser::Operator::I16x8GeU
            | wasmparser::Operator::I32x4Eq
            | wasmparser::Operator::I32x4Ne
            | wasmparser::Operator::I32x4LtS
            | wasmparser::Operator::I32x4LtU
            | wasmparser::Operator::I32x4GtS
            | wasmparser::Operator::I32x4GtU
            | wasmparser::Operator::I32x4LeS
            | wasmparser::Operator::I32x4LeU
            | wasmparser::Operator::I32x4GeS
            | wasmparser::Operator::I32x4GeU
            | wasmparser::Operator::I64x2Eq
            | wasmparser::Operator::I64x2Ne
            | wasmparser::Operator::I64x2LtS
            | wasmparser::Operator::I64x2GtS
            | wasmparser::Operator::I64x2LeS
            | wasmparser::Operator::I64x2GeS => Ok(None),
            // deterministic float operation
            wasmparser::Operator::F32x4Eq
            | wasmparser::Operator::F32x4Ne
            | wasmparser::Operator::F32x4Lt
            | wasmparser::Operator::F32x4Gt
            | wasmparser::Operator::F32x4Le
            | wasmparser::Operator::F32x4Ge
            | wasmparser::Operator::F64x2Eq
            | wasmparser::Operator::F64x2Ne
            | wasmparser::Operator::F64x2Lt
            | wasmparser::Operator::F64x2Gt
            | wasmparser::Operator::F64x2Le
            | wasmparser::Operator::F64x2Ge => Ok(None),
            // non-float operation
            wasmparser::Operator::V128Not
            | wasmparser::Operator::V128And
            | wasmparser::Operator::V128AndNot
            | wasmparser::Operator::V128Or
            | wasmparser::Operator::V128Xor
            | wasmparser::Operator::V128Bitselect
            | wasmparser::Operator::V128AnyTrue
            | wasmparser::Operator::I8x16Abs
            | wasmparser::Operator::I8x16Neg
            | wasmparser::Operator::I8x16Popcnt
            | wasmparser::Operator::I8x16AllTrue
            | wasmparser::Operator::I8x16Bitmask
            | wasmparser::Operator::I8x16NarrowI16x8S
            | wasmparser::Operator::I8x16NarrowI16x8U
            | wasmparser::Operator::I8x16Shl
            | wasmparser::Operator::I8x16ShrS
            | wasmparser::Operator::I8x16ShrU
            | wasmparser::Operator::I8x16Add
            | wasmparser::Operator::I8x16AddSatS
            | wasmparser::Operator::I8x16AddSatU
            | wasmparser::Operator::I8x16Sub
            | wasmparser::Operator::I8x16SubSatS
            | wasmparser::Operator::I8x16SubSatU
            | wasmparser::Operator::I8x16MinS
            | wasmparser::Operator::I8x16MinU
            | wasmparser::Operator::I8x16MaxS
            | wasmparser::Operator::I8x16MaxU
            | wasmparser::Operator::I8x16AvgrU
            | wasmparser::Operator::I16x8ExtAddPairwiseI8x16S
            | wasmparser::Operator::I16x8ExtAddPairwiseI8x16U
            | wasmparser::Operator::I16x8Abs
            | wasmparser::Operator::I16x8Neg
            | wasmparser::Operator::I16x8Q15MulrSatS
            | wasmparser::Operator::I16x8AllTrue
            | wasmparser::Operator::I16x8Bitmask
            | wasmparser::Operator::I16x8NarrowI32x4S
            | wasmparser::Operator::I16x8NarrowI32x4U
            | wasmparser::Operator::I16x8ExtendLowI8x16S
            | wasmparser::Operator::I16x8ExtendHighI8x16S
            | wasmparser::Operator::I16x8ExtendLowI8x16U
            | wasmparser::Operator::I16x8ExtendHighI8x16U
            | wasmparser::Operator::I16x8Shl
            | wasmparser::Operator::I16x8ShrS
            | wasmparser::Operator::I16x8ShrU
            | wasmparser::Operator::I16x8Add
            | wasmparser::Operator::I16x8AddSatS
            | wasmparser::Operator::I16x8AddSatU
            | wasmparser::Operator::I16x8Sub
            | wasmparser::Operator::I16x8SubSatS
            | wasmparser::Operator::I16x8SubSatU
            | wasmparser::Operator::I16x8Mul
            | wasmparser::Operator::I16x8MinS
            | wasmparser::Operator::I16x8MinU
            | wasmparser::Operator::I16x8MaxS
            | wasmparser::Operator::I16x8MaxU
            | wasmparser::Operator::I16x8AvgrU
            | wasmparser::Operator::I16x8ExtMulLowI8x16S
            | wasmparser::Operator::I16x8ExtMulHighI8x16S
            | wasmparser::Operator::I16x8ExtMulLowI8x16U
            | wasmparser::Operator::I16x8ExtMulHighI8x16U
            | wasmparser::Operator::I32x4ExtAddPairwiseI16x8S
            | wasmparser::Operator::I32x4ExtAddPairwiseI16x8U
            | wasmparser::Operator::I32x4Abs
            | wasmparser::Operator::I32x4Neg
            | wasmparser::Operator::I32x4AllTrue
            | wasmparser::Operator::I32x4Bitmask
            | wasmparser::Operator::I32x4ExtendLowI16x8S
            | wasmparser::Operator::I32x4ExtendHighI16x8S
            | wasmparser::Operator::I32x4ExtendLowI16x8U
            | wasmparser::Operator::I32x4ExtendHighI16x8U
            | wasmparser::Operator::I32x4Shl
            | wasmparser::Operator::I32x4ShrS
            | wasmparser::Operator::I32x4ShrU
            | wasmparser::Operator::I32x4Add
            | wasmparser::Operator::I32x4Sub
            | wasmparser::Operator::I32x4Mul
            | wasmparser::Operator::I32x4MinS
            | wasmparser::Operator::I32x4MinU
            | wasmparser::Operator::I32x4MaxS
            | wasmparser::Operator::I32x4MaxU
            | wasmparser::Operator::I32x4DotI16x8S
            | wasmparser::Operator::I32x4ExtMulLowI16x8S
            | wasmparser::Operator::I32x4ExtMulHighI16x8S
            | wasmparser::Operator::I32x4ExtMulLowI16x8U
            | wasmparser::Operator::I32x4ExtMulHighI16x8U
            | wasmparser::Operator::I64x2Abs
            | wasmparser::Operator::I64x2Neg
            | wasmparser::Operator::I64x2AllTrue
            | wasmparser::Operator::I64x2Bitmask
            | wasmparser::Operator::I64x2ExtendLowI32x4S
            | wasmparser::Operator::I64x2ExtendHighI32x4S
            | wasmparser::Operator::I64x2ExtendLowI32x4U
            | wasmparser::Operator::I64x2ExtendHighI32x4U
            | wasmparser::Operator::I64x2Shl
            | wasmparser::Operator::I64x2ShrS
            | wasmparser::Operator::I64x2ShrU
            | wasmparser::Operator::I64x2Add
            | wasmparser::Operator::I64x2Sub
            | wasmparser::Operator::I64x2Mul
            | wasmparser::Operator::I64x2ExtMulLowI32x4S
            | wasmparser::Operator::I64x2ExtMulHighI32x4S
            | wasmparser::Operator::I64x2ExtMulLowI32x4U
            | wasmparser::Operator::I64x2ExtMulHighI32x4U => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F32x4Ceil
            | wasmparser::Operator::F32x4Floor
            | wasmparser::Operator::F32x4Trunc
            | wasmparser::Operator::F32x4Nearest => Ok(Some(MaybeNaNKind::F32x4)),
            // deterministic float operation
            wasmparser::Operator::F32x4Abs | wasmparser::Operator::F32x4Neg => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F32x4Sqrt
            | wasmparser::Operator::F32x4Add
            | wasmparser::Operator::F32x4Sub
            | wasmparser::Operator::F32x4Mul
            | wasmparser::Operator::F32x4Div
            | wasmparser::Operator::F32x4Min
            | wasmparser::Operator::F32x4Max => Ok(Some(MaybeNaNKind::F32x4)),
            // deterministic float operation
            wasmparser::Operator::F32x4PMin | wasmparser::Operator::F32x4PMax => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F64x2Ceil
            | wasmparser::Operator::F64x2Floor
            | wasmparser::Operator::F64x2Trunc
            | wasmparser::Operator::F64x2Nearest => Ok(Some(MaybeNaNKind::F64x2)),
            // deterministic float operation
            wasmparser::Operator::F64x2Abs | wasmparser::Operator::F64x2Neg => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F64x2Sqrt
            | wasmparser::Operator::F64x2Add
            | wasmparser::Operator::F64x2Sub
            | wasmparser::Operator::F64x2Mul
            | wasmparser::Operator::F64x2Div
            | wasmparser::Operator::F64x2Min
            | wasmparser::Operator::F64x2Max => Ok(Some(MaybeNaNKind::F64x2)),
            // deterministic float operation
            wasmparser::Operator::F64x2PMin | wasmparser::Operator::F64x2PMax => Ok(None),
            // truncate float to int, which saturates to be deterministic
            wasmparser::Operator::I32x4TruncSatF32x4S
            | wasmparser::Operator::I32x4TruncSatF32x4U => Ok(None),
            // convert int to float, deterministic
            wasmparser::Operator::F32x4ConvertI32x4S | wasmparser::Operator::F32x4ConvertI32x4U => {
                Ok(None)
            }
            // truncate float to int, which saturates to be deterministic
            wasmparser::Operator::I32x4TruncSatF64x2SZero
            | wasmparser::Operator::I32x4TruncSatF64x2UZero => Ok(None),
            // convert int to float, deterministic
            wasmparser::Operator::F64x2ConvertLowI32x4S
            | wasmparser::Operator::F64x2ConvertLowI32x4U => Ok(None),
            // non-deterministic float operation that may produce any NaN
            wasmparser::Operator::F32x4DemoteF64x2Zero => Ok(Some(MaybeNaNKind::F32x4)),
            wasmparser::Operator::F64x2PromoteLowF32x4 => Ok(Some(MaybeNaNKind::F64x2)),
            // === Relaxed SIMD ===
            // non-deterministic, result may be platform-dependent
            wasmparser::Operator::I8x16RelaxedSwizzle
            | wasmparser::Operator::I32x4RelaxedTruncF32x4S
            | wasmparser::Operator::I32x4RelaxedTruncF32x4U
            | wasmparser::Operator::I32x4RelaxedTruncF64x2SZero
            | wasmparser::Operator::I32x4RelaxedTruncF64x2UZero
            | wasmparser::Operator::F32x4RelaxedMadd
            | wasmparser::Operator::F32x4RelaxedNmadd
            | wasmparser::Operator::F64x2RelaxedMadd
            | wasmparser::Operator::F64x2RelaxedNmadd
            | wasmparser::Operator::I8x16RelaxedLaneselect
            | wasmparser::Operator::I16x8RelaxedLaneselect
            | wasmparser::Operator::I32x4RelaxedLaneselect
            | wasmparser::Operator::I64x2RelaxedLaneselect
            | wasmparser::Operator::F32x4RelaxedMin
            | wasmparser::Operator::F32x4RelaxedMax
            | wasmparser::Operator::F64x2RelaxedMin
            | wasmparser::Operator::F64x2RelaxedMax
            | wasmparser::Operator::I16x8RelaxedQ15mulrS
            | wasmparser::Operator::I16x8RelaxedDotI8x16I7x16S
            | wasmparser::Operator::I32x4RelaxedDotI8x16I7x16AddS => {
                Err(NonDeterministicWasmFeature::RelaxedSimd)
            }
            // === Typed function references ===
            // non-float operation
            wasmparser::Operator::CallRef { .. }
            | wasmparser::Operator::ReturnCallRef { .. }
            | wasmparser::Operator::RefAsNonNull
            | wasmparser::Operator::BrOnNull { .. }
            | wasmparser::Operator::BrOnNonNull { .. } => Ok(None),
            // === Stack switching ===
            // non-float operation
            wasmparser::Operator::ContNew { .. }
            | wasmparser::Operator::ContBind { .. }
            | wasmparser::Operator::Suspend { .. }
            | wasmparser::Operator::Resume { .. }
            | wasmparser::Operator::ResumeThrow { .. }
            | wasmparser::Operator::Switch { .. } => Ok(None),
            // === Wide Arithmetic ===
            // non-float operation
            wasmparser::Operator::I64Add128
            | wasmparser::Operator::I64Sub128
            | wasmparser::Operator::I64MulWideS
            | wasmparser::Operator::I64MulWideU => Ok(None),
            // === FIXME ===
            #[cfg(not(test))]
            #[expect(clippy::panic)]
            _ => panic!("unsupported instruction"),
            #[cfg(test)]
            #[expect(unsafe_code)]
            _ => {
                extern "C" {
                    fn nan_canonicaliser_unhandled_operator() -> !;
                }
                unsafe { nan_canonicaliser_unhandled_operator() }
            }
        }
    }
}

#[derive(Copy, Clone)]
enum MaybeNaNKind {
    F32,
    F64,
    F32x4,
    F64x2,
}

#[derive(Debug, thiserror::Error)]
enum NonDeterministicWasmFeature {
    #[error(
        "WASM uses the non-deterministic relaxed-simd feature, which may produce \
         platform-dependent results"
    )]
    RelaxedSimd,
    #[error("WASM uses the non-deterministic threads feature, which may produce data races")]
    Threads,
    #[error(
        "WASM uses the non-deterministic shared-everything threads feature, which may produce \
         data races"
    )]
    SharedEverythingThreads,
}
