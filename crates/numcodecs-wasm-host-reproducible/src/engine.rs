use wasm_runtime_layer::{
    backend::{
        AsContext, AsContextMut, Export, Extern, Imports, Value, WasmEngine, WasmExternRef,
        WasmFunc, WasmGlobal, WasmInstance, WasmMemory, WasmModule, WasmStore, WasmStoreContext,
        WasmStoreContextMut, WasmTable,
    },
    ExportType, ExternType, FuncType, GlobalType, ImportType, MemoryType, TableType,
};

use crate::transform::{
    instcnt::{InstructionCounterInjecter, PerfWitInterfaces},
    nan::NaNCanonicaliser,
};

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedEngine<E: WasmEngine>(E);

impl<E: WasmEngine> WasmEngine for ValidatedEngine<E> {
    type ExternRef = ValidatedExternRef<E>;
    type Func = ValidatedFunc<E>;
    type Global = ValidatedGlobal<E>;
    type Instance = ValidatedInstance<E>;
    type Memory = ValidatedMemory<E>;
    type Module = ValidatedModule<E>;
    type Store<T> = ValidatedStore<T, E>;
    type StoreContext<'a, T: 'a> = ValidatedStoreContext<'a, T, E>;
    type StoreContextMut<'a, T: 'a> = ValidatedStoreContextMut<'a, T, E>;
    type Table = ValidatedTable<E>;
}

impl<E: WasmEngine> ValidatedEngine<E> {
    pub const fn new(engine: E) -> Self {
        Self(engine)
    }

    const fn as_ref(&self) -> &E {
        &self.0
    }

    const fn from_ref(engine: &E) -> &Self {
        // Safety: Self is a transparent newtype around E
        #[expect(unsafe_code)]
        unsafe {
            &*std::ptr::from_ref(engine).cast()
        }
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedExternRef<E: WasmEngine>(E::ExternRef);

impl<E: WasmEngine> WasmExternRef<ValidatedEngine<E>> for ValidatedExternRef<E> {
    fn new<T: 'static + Send + Sync>(
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        object: T,
    ) -> Self {
        Self(<E::ExternRef as WasmExternRef<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            object,
        ))
    }

    fn downcast<'a, 's: 'a, T: 'static, S: 'a>(
        &'a self,
        store: ValidatedStoreContext<'s, S, E>,
    ) -> anyhow::Result<&'a T> {
        WasmExternRef::downcast(&self.0, store.0)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedFunc<E: WasmEngine>(E::Func);

impl<E: WasmEngine> WasmFunc<ValidatedEngine<E>> for ValidatedFunc<E> {
    fn new<T>(
        mut ctx: impl AsContextMut<ValidatedEngine<E>, UserState = T>,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(
                ValidatedStoreContextMut<T, E>,
                &[Value<ValidatedEngine<E>>],
                &mut [Value<ValidatedEngine<E>>],
            ) -> anyhow::Result<()>,
    ) -> Self {
        Self(<E::Func as WasmFunc<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            ty,
            move |ctx, args, results| {
                func(
                    ValidatedStoreContextMut(ctx),
                    from_values(args),
                    from_values_mut(results),
                )
            },
        ))
    }

    fn ty(&self, ctx: impl AsContext<ValidatedEngine<E>>) -> FuncType {
        WasmFunc::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        args: &[Value<ValidatedEngine<E>>],
        results: &mut [Value<ValidatedEngine<E>>],
    ) -> anyhow::Result<()> {
        WasmFunc::call::<T>(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            as_values(args),
            as_values_mut(results),
        )
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedGlobal<E: WasmEngine>(E::Global);

impl<E: WasmEngine> WasmGlobal<ValidatedEngine<E>> for ValidatedGlobal<E> {
    fn new(
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        value: Value<ValidatedEngine<E>>,
        mutable: bool,
    ) -> Self {
        Self(<E::Global as WasmGlobal<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            into_value(value),
            mutable,
        ))
    }

    fn ty(&self, ctx: impl AsContext<ValidatedEngine<E>>) -> GlobalType {
        WasmGlobal::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        new_value: Value<ValidatedEngine<E>>,
    ) -> anyhow::Result<()> {
        WasmGlobal::set(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            into_value(new_value),
        )
    }

    fn get(&self, mut ctx: impl AsContextMut<ValidatedEngine<E>>) -> Value<ValidatedEngine<E>> {
        from_value(WasmGlobal::get(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
        ))
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedInstance<E: WasmEngine>(E::Instance);

impl<E: WasmEngine> WasmInstance<ValidatedEngine<E>> for ValidatedInstance<E> {
    fn new(
        mut store: impl AsContextMut<ValidatedEngine<E>>,
        module: &ValidatedModule<E>,
        imports: &Imports<ValidatedEngine<E>>,
    ) -> anyhow::Result<Self> {
        let mut new_imports = Imports::new();
        new_imports.extend(
            imports
                .into_iter()
                .map(|((module, name), value)| ((module, name), into_extern(value))),
        );

        let PerfWitInterfaces {
            perf: perf_interface,
            instruction_counter,
        } = PerfWitInterfaces::get();
        new_imports.define(
            &format!("{perf_interface}"),
            instruction_counter,
            Extern::Global(
                store
                    .as_context_mut()
                    .get_instruction_counter_global()
                    .0
                    .clone(),
            ),
        );

        Ok(Self(<E::Instance as WasmInstance<E>>::new(
            store.as_context_mut().as_inner_context_mut(),
            &module.0,
            &new_imports,
        )?))
    }

    fn exports(
        &self,
        store: impl AsContext<ValidatedEngine<E>>,
    ) -> Box<dyn Iterator<Item = Export<ValidatedEngine<E>>>> {
        Box::new(
            WasmInstance::exports(&self.0, store.as_context().as_inner_context()).map(
                |Export { name, value }| Export {
                    name,
                    value: from_extern(value),
                },
            ),
        )
    }

    fn get_export(
        &self,
        store: impl AsContext<ValidatedEngine<E>>,
        name: &str,
    ) -> Option<Extern<ValidatedEngine<E>>> {
        WasmInstance::get_export(&self.0, store.as_context().as_inner_context(), name)
            .map(from_extern)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedMemory<E: WasmEngine>(E::Memory);

impl<E: WasmEngine> WasmMemory<ValidatedEngine<E>> for ValidatedMemory<E> {
    fn new(mut ctx: impl AsContextMut<ValidatedEngine<E>>, ty: MemoryType) -> anyhow::Result<Self> {
        Ok(Self(<E::Memory as WasmMemory<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            ty,
        )?))
    }

    fn ty(&self, ctx: impl AsContext<ValidatedEngine<E>>) -> MemoryType {
        WasmMemory::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        additional: u32,
    ) -> anyhow::Result<u32> {
        WasmMemory::grow(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            additional,
        )
    }

    fn current_pages(&self, ctx: impl AsContext<ValidatedEngine<E>>) -> u32 {
        WasmMemory::current_pages(&self.0, ctx.as_context().as_inner_context())
    }

    fn read(
        &self,
        ctx: impl AsContext<ValidatedEngine<E>>,
        offset: usize,
        buffer: &mut [u8],
    ) -> anyhow::Result<()> {
        WasmMemory::read(&self.0, ctx.as_context().as_inner_context(), offset, buffer)
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        offset: usize,
        buffer: &[u8],
    ) -> anyhow::Result<()> {
        WasmMemory::write(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            offset,
            buffer,
        )
    }
}

pub const DETERMINISTIC_WASM_MODULE_FEATURES: wasmparser::WasmFeaturesInflated =
    wasmparser::WasmFeaturesInflated {
        // MUST: mutable globals do not introduce non-determinism, as long
        //       as the host does not change their value to be non-
        //       deterministic
        mutable_global: true,
        // OK: saturating float -> int conversions only produce finite values
        saturating_float_to_int: true,
        // MUST: arithmetic sign extension operators are deterministic
        sign_extension: true,
        // (unsure): disabled for now, needs further research
        reference_types: false,
        // OK: returning multiple values does not interact with determinism
        multi_value: true,
        // MUST: operations like memcpy and memset are deterministic
        bulk_memory: true,
        // (ok): fixed-width SIMD replicates scalar float semantics
        simd: true,
        // BAD: exposes platform-dependent behaviour and non-determinism
        relaxed_simd: false,
        // BAD: allows non-deterministic concurrency and race conditions
        threads: false,
        // BAD: allows non-deterministic concurrency and race conditions
        shared_everything_threads: false,
        // (ok): using tail calls does not interact with determinism
        //       but support is not universal yet:
        //       https://webassembly.org/features/
        tail_call: false,
        // BAD: float operations can introduce non-deterministic NaNs
        floats: false,
        // MUST: using multiple memories does not interact with determinism
        multi_memory: true,
        // (unsure): disabled for now, needs further research
        exceptions: false,
        // (nope): using a 64bit memory space does not interact with
        //         determinism but encourages large memory usage
        memory64: false,
        // (ok): const i[32|64] add, sub, and mul are deterministic
        //       but support is not universal yet:
        //       https://webassembly.org/features/
        extended_const: false,
        // NO-CORE: components must have been translated into core WASM
        //          modules by now
        component_model: false,
        // (unsure): disabled for now, needs further research
        function_references: false,
        // (unsure): disabled for now, needs further research
        memory_control: false,
        // (unsure): disabled for now, needs further research
        gc: false,
        // (ok): statically declaring a custom page size is deterministic
        //       and could reduce resource consumption
        //       but there is no support yet
        custom_page_sizes: false,
        // NO-CORE: components must have been translated into core WASM
        //          modules by now
        component_model_values: false,
        // NO-CORE: components must have been translated into core WASM
        //          modules by now
        component_model_nested_names: false,
        // NO-CORE: components must have been translated into core WASM
        //          modules by now
        component_model_more_flags: false,
        // NO-CORE: components must have been translated into core WASM
        //          modules by now
        component_model_multiple_returns: false,
        // (unsure): disabled for now, needs further research
        legacy_exceptions: false,
        // (unsure): disabled for now, depends on reference types and gc,
        //           needs further research
        gc_types: false,
        // (unsure): disabled for now, not needed since codecs are sync for now
        stack_switching: false,
        // OK: wide integer add, sub, and mul are deterministic
        wide_arithmetic: true,
        // NO-CORE: components must have been translated into core WASM
        //          modules by now
        component_model_async: false,
    };

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedModule<E: WasmEngine>(E::Module);

impl<E: WasmEngine> WasmModule<ValidatedEngine<E>> for ValidatedModule<E> {
    fn new(engine: &ValidatedEngine<E>, mut stream: impl std::io::Read) -> anyhow::Result<Self> {
        let features = wasmparser::WasmFeatures::from(wasmparser::WasmFeaturesInflated {
            // MUST: floats are required and we are running the NaN
            //       canonicalisation transform to make them deterministic
            floats: true,
            ..DETERMINISTIC_WASM_MODULE_FEATURES
        });

        let mut bytes = Vec::new();
        stream.read_to_end(&mut bytes)?;

        wasmparser::Validator::new_with_features(features).validate_all(&bytes)?;

        // Inject an instruction counter into the WASM module
        let bytes = InstructionCounterInjecter::apply_to_module(&bytes, features)?;

        // Normalise NaNs to ensure floating point operations are deterministic
        let bytes = NaNCanonicaliser::apply_to_module(&bytes, features)?;

        Ok(Self(<E::Module as WasmModule<E>>::new(
            engine.as_ref(),
            bytes.as_slice(),
        )?))
    }

    fn exports(&self) -> Box<dyn '_ + Iterator<Item = ExportType<'_>>> {
        WasmModule::exports(&self.0)
    }

    fn get_export(&self, name: &str) -> Option<ExternType> {
        WasmModule::get_export(&self.0, name)
    }

    fn imports(&self) -> Box<dyn '_ + Iterator<Item = ImportType<'_>>> {
        WasmModule::imports(&self.0)
    }
}

struct StoreData<T, E: WasmEngine> {
    data: T,
    instruction_counter: Option<ValidatedGlobal<E>>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedStore<T, E: WasmEngine>(E::Store<StoreData<T, E>>);

impl<T, E: WasmEngine> WasmStore<T, ValidatedEngine<E>> for ValidatedStore<T, E> {
    fn new(engine: &ValidatedEngine<E>, data: T) -> Self {
        Self(<E::Store<StoreData<T, E>> as WasmStore<
            StoreData<T, E>,
            E,
        >>::new(
            engine.as_ref(),
            StoreData {
                data,
                instruction_counter: None,
            },
        ))
    }

    fn engine(&self) -> &ValidatedEngine<E> {
        ValidatedEngine::from_ref(WasmStore::engine(&self.0))
    }

    fn data(&self) -> &T {
        &WasmStore::data(&self.0).data
    }

    fn data_mut(&mut self) -> &mut T {
        &mut WasmStore::data_mut(&mut self.0).data
    }

    fn into_data(self) -> T {
        WasmStore::into_data(self.0).data
    }
}

impl<T, E: WasmEngine> AsContext<ValidatedEngine<E>> for ValidatedStore<T, E> {
    type UserState = T;

    fn as_context(&self) -> ValidatedStoreContext<'_, Self::UserState, E> {
        ValidatedStoreContext(AsContext::as_context(&self.0))
    }
}

impl<T, E: WasmEngine> AsContextMut<ValidatedEngine<E>> for ValidatedStore<T, E> {
    fn as_context_mut(&mut self) -> ValidatedStoreContextMut<'_, Self::UserState, E> {
        ValidatedStoreContextMut(AsContextMut::as_context_mut(&mut self.0))
    }
}

#[repr(transparent)]
pub struct ValidatedStoreContext<'a, T: 'a, E: WasmEngine>(E::StoreContext<'a, StoreData<T, E>>);

impl<'a, T: 'a, E: WasmEngine> WasmStoreContext<'a, T, ValidatedEngine<E>>
    for ValidatedStoreContext<'a, T, E>
{
    fn engine(&self) -> &ValidatedEngine<E> {
        ValidatedEngine::from_ref(WasmStoreContext::engine(&self.0))
    }

    fn data(&self) -> &T {
        &WasmStoreContext::data(&self.0).data
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContext<ValidatedEngine<E>> for ValidatedStoreContext<'a, T, E> {
    type UserState = T;

    fn as_context(&self) -> ValidatedStoreContext<'_, Self::UserState, E> {
        ValidatedStoreContext(AsContext::as_context(&self.0))
    }
}

impl<'a, T: 'a, E: WasmEngine> ValidatedStoreContext<'a, T, E> {
    fn as_inner_context(&self) -> E::StoreContext<'_, StoreData<T, E>> {
        self.0.as_context()
    }
}

#[repr(transparent)]
pub struct ValidatedStoreContextMut<'a, T: 'a, E: WasmEngine>(
    E::StoreContextMut<'a, StoreData<T, E>>,
);

impl<'a, T: 'a, E: WasmEngine> WasmStoreContext<'a, T, ValidatedEngine<E>>
    for ValidatedStoreContextMut<'a, T, E>
{
    fn engine(&self) -> &ValidatedEngine<E> {
        ValidatedEngine::from_ref(WasmStoreContext::engine(&self.0))
    }

    fn data(&self) -> &T {
        &WasmStoreContext::data(&self.0).data
    }
}

impl<'a, T: 'a, E: WasmEngine> WasmStoreContextMut<'a, T, ValidatedEngine<E>>
    for ValidatedStoreContextMut<'a, T, E>
{
    fn data_mut(&mut self) -> &mut T {
        &mut WasmStoreContextMut::data_mut(&mut self.0).data
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContext<ValidatedEngine<E>>
    for ValidatedStoreContextMut<'a, T, E>
{
    type UserState = T;

    fn as_context(&self) -> ValidatedStoreContext<'_, Self::UserState, E> {
        ValidatedStoreContext(AsContext::as_context(&self.0))
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContextMut<ValidatedEngine<E>>
    for ValidatedStoreContextMut<'a, T, E>
{
    fn as_context_mut(&mut self) -> ValidatedStoreContextMut<'_, Self::UserState, E> {
        ValidatedStoreContextMut(AsContextMut::as_context_mut(&mut self.0))
    }
}

impl<'a, T: 'a, E: WasmEngine> ValidatedStoreContextMut<'a, T, E> {
    fn as_inner_context_mut(&mut self) -> E::StoreContextMut<'_, StoreData<T, E>> {
        self.0.as_context_mut()
    }

    fn get_instruction_counter_global(&mut self) -> &ValidatedGlobal<E> {
        let mut this = self;

        // NLL cannot prove this to be safe, but Polonius can
        polonius_the_crab::polonius!(|this| -> &'polonius ValidatedGlobal<E> {
            let data: &mut StoreData<T, E> = WasmStoreContextMut::data_mut(&mut this.0);
            if let Some(global) = &data.instruction_counter {
                polonius_the_crab::polonius_return!(global);
            }
        });

        let global = WasmGlobal::new(AsContextMut::as_context_mut(this), Value::I64(0), true);

        let data: &mut StoreData<T, E> = WasmStoreContextMut::data_mut(&mut this.0);
        data.instruction_counter.insert(global)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ValidatedTable<E: WasmEngine>(E::Table);

impl<E: WasmEngine> WasmTable<ValidatedEngine<E>> for ValidatedTable<E> {
    fn new(
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        ty: TableType,
        init: Value<ValidatedEngine<E>>,
    ) -> anyhow::Result<Self> {
        Ok(Self(<E::Table as WasmTable<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            ty,
            into_value(init),
        )?))
    }

    fn ty(&self, ctx: impl AsContext<ValidatedEngine<E>>) -> TableType {
        WasmTable::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn size(&self, ctx: impl AsContext<ValidatedEngine<E>>) -> u32 {
        WasmTable::size(&self.0, ctx.as_context().as_inner_context())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        delta: u32,
        init: Value<ValidatedEngine<E>>,
    ) -> anyhow::Result<u32> {
        WasmTable::grow(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            delta,
            into_value(init),
        )
    }

    fn get(
        &self,
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        index: u32,
    ) -> Option<Value<ValidatedEngine<E>>> {
        WasmTable::get(&self.0, ctx.as_context_mut().as_inner_context_mut(), index).map(from_value)
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<ValidatedEngine<E>>,
        index: u32,
        value: Value<ValidatedEngine<E>>,
    ) -> anyhow::Result<()> {
        WasmTable::set(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            index,
            into_value(value),
        )
    }
}

const fn as_values<E: WasmEngine>(values: &[Value<ValidatedEngine<E>>]) -> &[Value<E>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), values.len())
    }
}

fn as_values_mut<E: WasmEngine>(values: &mut [Value<ValidatedEngine<E>>]) -> &mut [Value<E>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len())
    }
}

const fn from_values<E: WasmEngine>(values: &[Value<E>]) -> &[Value<ValidatedEngine<E>>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), values.len())
    }
}

fn from_values_mut<E: WasmEngine>(values: &mut [Value<E>]) -> &mut [Value<ValidatedEngine<E>>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len())
    }
}

fn into_value<E: WasmEngine>(value: Value<ValidatedEngine<E>>) -> Value<E> {
    match value {
        Value::I32(v) => Value::I32(v),
        Value::I64(v) => Value::I64(v),
        Value::F32(v) => Value::F32(v),
        Value::F64(v) => Value::F64(v),
        Value::FuncRef(v) => Value::FuncRef(v.map(|v| v.0)),
        Value::ExternRef(v) => Value::ExternRef(v.map(|v| v.0)),
    }
}

fn from_value<E: WasmEngine>(value: Value<E>) -> Value<ValidatedEngine<E>> {
    match value {
        Value::I32(v) => Value::I32(v),
        Value::I64(v) => Value::I64(v),
        Value::F32(v) => Value::F32(v),
        Value::F64(v) => Value::F64(v),
        Value::FuncRef(v) => Value::FuncRef(v.map(ValidatedFunc)),
        Value::ExternRef(v) => Value::ExternRef(v.map(ValidatedExternRef)),
    }
}

fn into_extern<E: WasmEngine>(value: Extern<ValidatedEngine<E>>) -> Extern<E> {
    match value {
        Extern::Global(v) => Extern::Global(v.0),
        Extern::Table(v) => Extern::Table(v.0),
        Extern::Memory(v) => Extern::Memory(v.0),
        Extern::Func(v) => Extern::Func(v.0),
    }
}

fn from_extern<E: WasmEngine>(value: Extern<E>) -> Extern<ValidatedEngine<E>> {
    match value {
        Extern::Global(v) => Extern::Global(ValidatedGlobal(v)),
        Extern::Table(v) => Extern::Table(ValidatedTable(v)),
        Extern::Memory(v) => Extern::Memory(ValidatedMemory(v)),
        Extern::Func(v) => Extern::Func(ValidatedFunc(v)),
    }
}
