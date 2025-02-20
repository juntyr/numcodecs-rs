use std::sync::OnceLock;

use anyhow::{anyhow, Error};
use semver::Version;
use wasm_component_layer::{InterfaceIdentifier, PackageIdentifier, PackageName};
use wasm_encoder::reencode::{self, Reencode};

pub enum InstructionCounterInjecter {}

impl InstructionCounterInjecter {
    pub fn apply_to_module(
        wasm: &[u8],
        features: wasmparser::WasmFeatures,
    ) -> Result<Vec<u8>, anyhow::Error> {
        let mut parser = wasmparser::Parser::new(0);
        parser.set_features(features);

        let mut module = wasm_encoder::Module::new();

        let mut reencoder = InstructionCounterInjecterReencoder {
            instruction_counter_global: None,
            num_imported_funcs: 0,
            instruction_counter_func_index: None,
            func_index: 0,
        };
        reencoder
            .parse_core_module(&mut module, parser, wasm)
            .map_err(|err| anyhow::format_err!("{}", err))?;

        if let Some(instruction_counter_func_index) = reencoder.instruction_counter_func_index {
            anyhow::ensure!(
                reencoder.func_index > instruction_counter_func_index,
                "missing WASM instruction counter reader function body"
            );
        }

        let wasm = module.finish();
        wasmparser::Validator::new_with_features(features).validate_all(&wasm)?;
        Ok(wasm)
    }
}

struct InstructionCounterInjecterReencoder {
    instruction_counter_global: Option<u32>,
    num_imported_funcs: u32,
    instruction_counter_func_index: Option<u32>,
    func_index: u32,
}

impl wasm_encoder::reencode::Reencode for InstructionCounterInjecterReencoder {
    type Error = Error;

    fn global_index(&mut self, global: u32) -> u32 {
        match self.instruction_counter_global {
            Some(instruction_counter_global) if global >= instruction_counter_global => global + 1,
            _ => global,
        }
    }

    fn parse_import_section(
        &mut self,
        imports: &mut wasm_encoder::ImportSection,
        section: wasmparser::ImportSectionReader<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        self.instruction_counter_global.get_or_insert(0);
        for import in section {
            let import = import?;
            match import.ty {
                wasmparser::TypeRef::Func(_) => self.num_imported_funcs += 1,
                wasmparser::TypeRef::Global(_) => {
                    *self.instruction_counter_global.get_or_insert(0) += 1;
                }
                wasmparser::TypeRef::Table(_)
                | wasmparser::TypeRef::Memory(_)
                | wasmparser::TypeRef::Tag(_) => (),
            }
            self.parse_import(imports, import)?;
        }
        Self::inject_instruction_counter_import(imports);
        Ok(())
    }

    fn intersperse_section_hook(
        &mut self,
        module: &mut wasm_encoder::Module,
        _after: Option<wasm_encoder::SectionId>,
        before: Option<wasm_encoder::SectionId>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        // the function section directly follows the import section
        // if the function section is missing, we also don't need to do any
        //  instruction counting
        let Some(wasm_encoder::SectionId::Function) = before else {
            return Ok(());
        };

        if self.instruction_counter_global.is_none() {
            self.instruction_counter_global = Some(0);

            let mut imports = wasm_encoder::ImportSection::new();
            Self::inject_instruction_counter_import(&mut imports);
            module.section(&imports);
        }

        Ok(())
    }

    fn parse_export_section(
        &mut self,
        exports: &mut wasm_encoder::ExportSection,
        section: wasmparser::ExportSectionReader<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        let instruction_counter_export_name = {
            let PerfWitInterfaces {
                perf: perf_interface,
                instruction_counter,
                ..
            } = PerfWitInterfaces::get();

            format!("{perf_interface}#{instruction_counter}")
        };

        for export in section {
            let export = export?;
            if export.name == instruction_counter_export_name {
                if !matches!(export.kind, wasmparser::ExternalKind::Func) {
                    return Err(reencode::Error::UserError(anyhow!(
                        "instruction counter reader export must be a function",
                    )));
                }
                if self.instruction_counter_func_index.is_some() {
                    return Err(reencode::Error::UserError(anyhow!(
                        "duplicate instruction counter reader export",
                    )));
                }
                self.instruction_counter_func_index = Some(export.index - self.num_imported_funcs);
            }

            self.parse_export(exports, export);
        }
        Ok(())
    }

    fn parse_function_body(
        &mut self,
        code: &mut wasm_encoder::CodeSection,
        func: wasmparser::FunctionBody<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        let Some(instruction_counter_global) = self.instruction_counter_global else {
            return Err(reencode::Error::UserError(anyhow!(
                "missing instruction counter import",
            )));
        };

        let mut function = self.new_function_with_parsed_locals(&func)?;
        let instructions = func.get_operators_reader()?;

        if Some(self.func_index) == self.instruction_counter_func_index {
            let locals = func.get_locals_reader()?;

            if locals.get_count() > 0 {
                return Err(reencode::Error::UserError(anyhow!(
                    "instruction counter function has no locals",
                )));
            }
            let instructions = instructions.into_iter().collect::<Result<Vec<_>, _>>()?;
            if !matches!(
                instructions.as_slice(),
                [wasmparser::Operator::Unreachable, wasmparser::Operator::End]
            ) {
                return Err(reencode::Error::UserError(anyhow!(
                    "instruction counter function has a single instruction and is unreachable",
                )));
            }

            function.instruction(&wasm_encoder::Instruction::GlobalGet(
                instruction_counter_global,
            ));
            function.instruction(&wasm_encoder::Instruction::Return);
            function.instruction(&wasm_encoder::Instruction::End);
        } else {
            let mut counter: i64 = 0;

            for instruction in instructions {
                let instruction = instruction?;

                if let Some(update) = Self::instruction_needs_counter_update(&instruction) {
                    counter += 1;

                    if update {
                        for count_instruction in Self::generate_instruction_counter_update(
                            instruction_counter_global,
                            counter,
                        ) {
                            function.instruction(&count_instruction);
                        }
                    }

                    counter = 0;
                }

                function.instruction(&self.instruction(instruction)?);
            }
        }
        code.function(&function);
        self.func_index += 1;
        Ok(())
    }
}

impl InstructionCounterInjecterReencoder {
    fn inject_instruction_counter_import(imports: &mut wasm_encoder::ImportSection) {
        let PerfWitInterfaces {
            perf: perf_interface,
            instruction_counter,
        } = PerfWitInterfaces::get();

        imports.import(
            &format!("{perf_interface}"),
            instruction_counter,
            wasm_encoder::EntityType::Global(wasm_encoder::GlobalType {
                val_type: wasm_encoder::ValType::I64,
                mutable: true,
                shared: false,
            }),
        );
    }

    const fn generate_instruction_counter_update(
        instruction_counter_global: u32,
        delta: i64,
    ) -> impl IntoIterator<Item = wasm_encoder::Instruction<'static>> {
        [
            wasm_encoder::Instruction::GlobalGet(instruction_counter_global),
            wasm_encoder::Instruction::I64Const(delta),
            wasm_encoder::Instruction::I64Add,
            wasm_encoder::Instruction::GlobalSet(instruction_counter_global),
        ]
    }

    #[expect(clippy::too_many_lines)]
    fn instruction_needs_counter_update(instr: &wasmparser::Operator) -> Option<bool> {
        match instr {
            // === MVP ===
            // we cannot recover from an unreachable instruction, so instruction
            //  counting won't matter at that point anyways
            wasmparser::Operator::Unreachable => Some(false),
            // no-op is not an instruction
            wasmparser::Operator::Nop => None,
            // we need to save before diverging control flow, and these
            //  instructions are jump targets
            wasmparser::Operator::Block { .. }
            | wasmparser::Operator::Loop { .. }
            | wasmparser::Operator::If { .. }
            | wasmparser::Operator::Else => Some(true),
            // === Exception handling ===
            // we need to save before diverging control flow, and this
            //  instruction is a jump target
            wasmparser::Operator::TryTable { .. } => Some(true),
            // we need to save before diverging control flow
            wasmparser::Operator::Throw { .. } | wasmparser::Operator::ThrowRef => Some(true),
            // === Legacy exception handling (deprecated) ===
            // we need to save before diverging control flow, and this
            //  instruction is a jump target
            wasmparser::Operator::Try { .. } => Some(true),
            // we need to save before diverging control flow
            wasmparser::Operator::Catch { .. }
            | wasmparser::Operator::Rethrow { .. }
            | wasmparser::Operator::Delegate { .. }
            | wasmparser::Operator::CatchAll => Some(true),
            // === MVP ===
            // we conservatively save at the end of a scope
            wasmparser::Operator::End => Some(true),
            // we need to save before diverging control flow
            wasmparser::Operator::Br { .. }
            | wasmparser::Operator::BrIf { .. }
            | wasmparser::Operator::BrTable { .. } => Some(true),
            // we need to save before returning from a function
            wasmparser::Operator::Return => Some(true),
            // calling into a function will return control flow right back to
            //  here, so saving is not necessary
            wasmparser::Operator::Call { .. } | wasmparser::Operator::CallIndirect { .. } => {
                Some(false)
            }
            // === Tail calls ===
            // tail calls need to save since they return from this function
            wasmparser::Operator::ReturnCall { .. }
            | wasmparser::Operator::ReturnCallIndirect { .. } => Some(true),
            // === MVP ===
            // no control flow
            wasmparser::Operator::Drop | wasmparser::Operator::Select => Some(false),
            // === Reference types ===
            // no control flow
            wasmparser::Operator::TypedSelect { .. } => Some(false),
            // === MVP ===
            // no control flow
            wasmparser::Operator::LocalGet { .. }
            | wasmparser::Operator::LocalSet { .. }
            | wasmparser::Operator::LocalTee { .. }
            | wasmparser::Operator::GlobalGet { .. }
            | wasmparser::Operator::GlobalSet { .. }
            | wasmparser::Operator::I32Load { .. }
            | wasmparser::Operator::I64Load { .. }
            | wasmparser::Operator::F32Load { .. }
            | wasmparser::Operator::F64Load { .. }
            | wasmparser::Operator::I32Load8S { .. }
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
            | wasmparser::Operator::I64Store { .. }
            | wasmparser::Operator::F32Store { .. }
            | wasmparser::Operator::F64Store { .. }
            | wasmparser::Operator::I32Store8 { .. }
            | wasmparser::Operator::I32Store16 { .. }
            | wasmparser::Operator::I64Store8 { .. }
            | wasmparser::Operator::I64Store16 { .. }
            | wasmparser::Operator::I64Store32 { .. }
            | wasmparser::Operator::MemorySize { .. }
            | wasmparser::Operator::MemoryGrow { .. }
            | wasmparser::Operator::I32Const { .. }
            | wasmparser::Operator::I64Const { .. }
            | wasmparser::Operator::F32Const { .. }
            | wasmparser::Operator::F64Const { .. } => Some(false),
            // === Reference types ===
            // no control flow
            wasmparser::Operator::RefNull { .. }
            | wasmparser::Operator::RefIsNull
            | wasmparser::Operator::RefFunc { .. } => Some(false),
            // === Garbage collection ===
            // no control flow
            wasmparser::Operator::RefEq => Some(false),
            // === MVP ===
            // no control flow
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
            | wasmparser::Operator::I64GeU
            | wasmparser::Operator::F32Eq
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
            | wasmparser::Operator::F64Ge
            | wasmparser::Operator::I32Clz
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
            | wasmparser::Operator::I64Rotr
            | wasmparser::Operator::F32Abs
            | wasmparser::Operator::F32Neg
            | wasmparser::Operator::F32Ceil
            | wasmparser::Operator::F32Floor
            | wasmparser::Operator::F32Trunc
            | wasmparser::Operator::F32Nearest
            | wasmparser::Operator::F32Sqrt
            | wasmparser::Operator::F32Add
            | wasmparser::Operator::F32Sub
            | wasmparser::Operator::F32Mul
            | wasmparser::Operator::F32Div
            | wasmparser::Operator::F32Min
            | wasmparser::Operator::F32Max
            | wasmparser::Operator::F32Copysign
            | wasmparser::Operator::F64Abs
            | wasmparser::Operator::F64Neg
            | wasmparser::Operator::F64Ceil
            | wasmparser::Operator::F64Floor
            | wasmparser::Operator::F64Trunc
            | wasmparser::Operator::F64Nearest
            | wasmparser::Operator::F64Sqrt
            | wasmparser::Operator::F64Add
            | wasmparser::Operator::F64Sub
            | wasmparser::Operator::F64Mul
            | wasmparser::Operator::F64Div
            | wasmparser::Operator::F64Min
            | wasmparser::Operator::F64Max
            | wasmparser::Operator::F64Copysign
            | wasmparser::Operator::I32WrapI64
            | wasmparser::Operator::I32TruncF32S
            | wasmparser::Operator::I32TruncF32U
            | wasmparser::Operator::I32TruncF64S
            | wasmparser::Operator::I32TruncF64U
            | wasmparser::Operator::I64ExtendI32S
            | wasmparser::Operator::I64ExtendI32U
            | wasmparser::Operator::I64TruncF32S
            | wasmparser::Operator::I64TruncF32U
            | wasmparser::Operator::I64TruncF64S
            | wasmparser::Operator::I64TruncF64U
            | wasmparser::Operator::F32ConvertI32S
            | wasmparser::Operator::F32ConvertI32U
            | wasmparser::Operator::F32ConvertI64S
            | wasmparser::Operator::F32ConvertI64U
            | wasmparser::Operator::F32DemoteF64
            | wasmparser::Operator::F64ConvertI32S
            | wasmparser::Operator::F64ConvertI32U
            | wasmparser::Operator::F64ConvertI64S
            | wasmparser::Operator::F64ConvertI64U
            | wasmparser::Operator::F64PromoteF32
            | wasmparser::Operator::I32ReinterpretF32
            | wasmparser::Operator::I64ReinterpretF64
            | wasmparser::Operator::F32ReinterpretI32
            | wasmparser::Operator::F64ReinterpretI64 => Some(false),
            // === Sign extension ===
            // no control flow
            wasmparser::Operator::I32Extend8S
            | wasmparser::Operator::I32Extend16S
            | wasmparser::Operator::I64Extend8S
            | wasmparser::Operator::I64Extend16S
            | wasmparser::Operator::I64Extend32S => Some(false),
            // === Garbage collection ===
            // no control flow
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
            | wasmparser::Operator::ArrayInitElem { .. } => Some(false),
            // no control flow
            wasmparser::Operator::RefTestNonNull { .. }
            | wasmparser::Operator::RefTestNullable { .. } => Some(false),
            // no (observable) control flow (except for trapping)
            wasmparser::Operator::RefCastNonNull { .. }
            | wasmparser::Operator::RefCastNullable { .. } => Some(false),
            // we need to save before diverging control flow
            wasmparser::Operator::BrOnCast { .. } | wasmparser::Operator::BrOnCastFail { .. } => {
                Some(true)
            }
            // no control flow
            wasmparser::Operator::AnyConvertExtern
            | wasmparser::Operator::ExternConvertAny
            | wasmparser::Operator::RefI31
            | wasmparser::Operator::I31GetS
            | wasmparser::Operator::I31GetU => Some(false),
            // === Non-trapping float-to-int conversions ===
            // no control flow
            wasmparser::Operator::I32TruncSatF32S
            | wasmparser::Operator::I32TruncSatF32U
            | wasmparser::Operator::I32TruncSatF64S
            | wasmparser::Operator::I32TruncSatF64U
            | wasmparser::Operator::I64TruncSatF32S
            | wasmparser::Operator::I64TruncSatF32U
            | wasmparser::Operator::I64TruncSatF64S
            | wasmparser::Operator::I64TruncSatF64U => Some(false),
            // === Bulk memory ===
            // no control flow
            wasmparser::Operator::MemoryInit { .. }
            | wasmparser::Operator::DataDrop { .. }
            | wasmparser::Operator::MemoryCopy { .. }
            | wasmparser::Operator::MemoryFill { .. }
            | wasmparser::Operator::TableInit { .. }
            | wasmparser::Operator::ElemDrop { .. }
            | wasmparser::Operator::TableCopy { .. } => Some(false),
            // === Reference types ===
            // no control flow
            wasmparser::Operator::TableFill { .. }
            | wasmparser::Operator::TableGet { .. }
            | wasmparser::Operator::TableSet { .. }
            | wasmparser::Operator::TableGrow { .. }
            | wasmparser::Operator::TableSize { .. } => Some(false),
            // === Memory control ===
            // no control flow
            wasmparser::Operator::MemoryDiscard { .. } => Some(false),
            // === Threads ===
            // no control flow
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
            | wasmparser::Operator::I64AtomicRmw32CmpxchgU { .. } => Some(false),
            // === Shared-everything threads ===
            // no control flow
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
            | wasmparser::Operator::RefI31Shared => Some(false),
            // === SIMD ===
            // no control flow
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
            | wasmparser::Operator::V128Store64Lane { .. }
            | wasmparser::Operator::V128Const { .. }
            | wasmparser::Operator::I8x16Shuffle { .. }
            | wasmparser::Operator::I8x16ExtractLaneS { .. }
            | wasmparser::Operator::I8x16ExtractLaneU { .. }
            | wasmparser::Operator::I8x16ReplaceLane { .. }
            | wasmparser::Operator::I16x8ExtractLaneS { .. }
            | wasmparser::Operator::I16x8ExtractLaneU { .. }
            | wasmparser::Operator::I16x8ReplaceLane { .. }
            | wasmparser::Operator::I32x4ExtractLane { .. }
            | wasmparser::Operator::I32x4ReplaceLane { .. }
            | wasmparser::Operator::I64x2ExtractLane { .. }
            | wasmparser::Operator::I64x2ReplaceLane { .. }
            | wasmparser::Operator::F32x4ExtractLane { .. }
            | wasmparser::Operator::F32x4ReplaceLane { .. }
            | wasmparser::Operator::F64x2ExtractLane { .. }
            | wasmparser::Operator::F64x2ReplaceLane { .. }
            | wasmparser::Operator::I8x16Swizzle
            | wasmparser::Operator::I8x16Splat
            | wasmparser::Operator::I16x8Splat
            | wasmparser::Operator::I32x4Splat
            | wasmparser::Operator::I64x2Splat
            | wasmparser::Operator::F32x4Splat
            | wasmparser::Operator::F64x2Splat
            | wasmparser::Operator::I8x16Eq
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
            | wasmparser::Operator::I64x2GeS
            | wasmparser::Operator::F32x4Eq
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
            | wasmparser::Operator::F64x2Ge
            | wasmparser::Operator::V128Not
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
            | wasmparser::Operator::I64x2ExtMulHighI32x4U
            | wasmparser::Operator::F32x4Ceil
            | wasmparser::Operator::F32x4Floor
            | wasmparser::Operator::F32x4Trunc
            | wasmparser::Operator::F32x4Nearest
            | wasmparser::Operator::F32x4Abs
            | wasmparser::Operator::F32x4Neg
            | wasmparser::Operator::F32x4Sqrt
            | wasmparser::Operator::F32x4Add
            | wasmparser::Operator::F32x4Sub
            | wasmparser::Operator::F32x4Mul
            | wasmparser::Operator::F32x4Div
            | wasmparser::Operator::F32x4Min
            | wasmparser::Operator::F32x4Max
            | wasmparser::Operator::F32x4PMin
            | wasmparser::Operator::F32x4PMax
            | wasmparser::Operator::F64x2Ceil
            | wasmparser::Operator::F64x2Floor
            | wasmparser::Operator::F64x2Trunc
            | wasmparser::Operator::F64x2Nearest
            | wasmparser::Operator::F64x2Abs
            | wasmparser::Operator::F64x2Neg
            | wasmparser::Operator::F64x2Sqrt
            | wasmparser::Operator::F64x2Add
            | wasmparser::Operator::F64x2Sub
            | wasmparser::Operator::F64x2Mul
            | wasmparser::Operator::F64x2Div
            | wasmparser::Operator::F64x2Min
            | wasmparser::Operator::F64x2Max
            | wasmparser::Operator::F64x2PMin
            | wasmparser::Operator::F64x2PMax
            | wasmparser::Operator::I32x4TruncSatF32x4S
            | wasmparser::Operator::I32x4TruncSatF32x4U
            | wasmparser::Operator::F32x4ConvertI32x4S
            | wasmparser::Operator::F32x4ConvertI32x4U
            | wasmparser::Operator::I32x4TruncSatF64x2SZero
            | wasmparser::Operator::I32x4TruncSatF64x2UZero
            | wasmparser::Operator::F64x2ConvertLowI32x4S
            | wasmparser::Operator::F64x2ConvertLowI32x4U
            | wasmparser::Operator::F32x4DemoteF64x2Zero
            | wasmparser::Operator::F64x2PromoteLowF32x4 => Some(false),
            // === Relaxed SIMD ===
            // no control flow
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
            | wasmparser::Operator::I32x4RelaxedDotI8x16I7x16AddS => Some(false),
            // === Typed function references ===
            // calling into a typed function will return control flow right back
            //  back to here, so saving is not necessary
            wasmparser::Operator::CallRef { .. } => Some(false),
            // typed tail calls need to save since they return from this
            //  function
            wasmparser::Operator::ReturnCallRef { .. } => Some(true),
            // no (observable) control flow (except for trapping)
            wasmparser::Operator::RefAsNonNull => Some(false),
            // we need to save before diverging control flow
            wasmparser::Operator::BrOnNull { .. } | wasmparser::Operator::BrOnNonNull { .. } => {
                Some(true)
            }
            // === Stack switching ===
            // creating a continuation does not change the control flow (yet)
            wasmparser::Operator::ContNew { .. } | wasmparser::Operator::ContBind { .. } => {
                Some(false)
            }
            // we need to save before diverging control flow by suspending or
            //  resuming a continuation diverg
            wasmparser::Operator::Suspend { .. }
            | wasmparser::Operator::Resume { .. }
            | wasmparser::Operator::ResumeThrow { .. } => Some(true),
            // we need to save before diverging control flow by switching to a
            //  different continuation
            wasmparser::Operator::Switch { .. } => Some(true),
            // === Wide Arithmetic ===
            // no control flow
            wasmparser::Operator::I64Add128
            | wasmparser::Operator::I64Sub128
            | wasmparser::Operator::I64MulWideS
            | wasmparser::Operator::I64MulWideU => Some(false),
            // === FIXME ===
            #[cfg(not(test))]
            #[expect(clippy::panic)]
            _ => panic!("unsupported instruction"),
            #[cfg(test)]
            #[expect(unsafe_code)]
            _ => {
                extern "C" {
                    fn instruction_counter_unhandled_operator() -> !;
                }
                unsafe { instruction_counter_unhandled_operator() }
            }
        }
    }
}

pub struct PerfWitInterfaces {
    pub perf: InterfaceIdentifier,
    pub instruction_counter: String,
}

impl PerfWitInterfaces {
    #[must_use]
    pub fn get() -> &'static Self {
        static PERF_WIT_INTERFACES: OnceLock<PerfWitInterfaces> = OnceLock::new();

        PERF_WIT_INTERFACES.get_or_init(|| Self {
            perf: InterfaceIdentifier::new(
                PackageIdentifier::new(
                    PackageName::new("numcodecs", "wasm"),
                    Some(Version::new(0, 1, 0)),
                ),
                "perf",
            ),
            instruction_counter: String::from("instruction-counter"),
        })
    }
}
