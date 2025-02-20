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
pub struct ReproducibleEngine<E: WasmEngine>(E);

impl<E: WasmEngine> WasmEngine for ReproducibleEngine<E> {
    type ExternRef = ReproducibleExternRef<E>;
    type Func = ReproducibleFunc<E>;
    type Global = ReproducibleGlobal<E>;
    type Instance = ReproducibleInstance<E>;
    type Memory = ReproducibleMemory<E>;
    type Module = ReproducibleModule<E>;
    type Store<T> = ReproducibleStore<T, E>;
    type StoreContext<'a, T: 'a> = ReproducibleStoreContext<'a, T, E>;
    type StoreContextMut<'a, T: 'a> = ReproducibleStoreContextMut<'a, T, E>;
    type Table = ReproducibleTable<E>;
}

impl<E: WasmEngine> ReproducibleEngine<E> {
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
pub struct ReproducibleExternRef<E: WasmEngine>(E::ExternRef);

impl<E: WasmEngine> WasmExternRef<ReproducibleEngine<E>> for ReproducibleExternRef<E> {
    fn new<T: 'static + Send + Sync>(
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        object: T,
    ) -> Self {
        Self(<E::ExternRef as WasmExternRef<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            object,
        ))
    }

    fn downcast<'a, 's: 'a, T: 'static, S: 'a>(
        &'a self,
        store: ReproducibleStoreContext<'s, S, E>,
    ) -> anyhow::Result<&'a T> {
        WasmExternRef::downcast(&self.0, store.0)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ReproducibleFunc<E: WasmEngine>(E::Func);

impl<E: WasmEngine> WasmFunc<ReproducibleEngine<E>> for ReproducibleFunc<E> {
    fn new<T>(
        mut ctx: impl AsContextMut<ReproducibleEngine<E>, UserState = T>,
        ty: FuncType,
        func: impl 'static
            + Send
            + Sync
            + Fn(
                ReproducibleStoreContextMut<T, E>,
                &[Value<ReproducibleEngine<E>>],
                &mut [Value<ReproducibleEngine<E>>],
            ) -> anyhow::Result<()>,
    ) -> Self {
        Self(<E::Func as WasmFunc<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            ty,
            move |ctx, args, results| {
                func(
                    ReproducibleStoreContextMut(ctx),
                    from_values(args),
                    from_values_mut(results),
                )
            },
        ))
    }

    fn ty(&self, ctx: impl AsContext<ReproducibleEngine<E>>) -> FuncType {
        WasmFunc::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn call<T>(
        &self,
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        args: &[Value<ReproducibleEngine<E>>],
        results: &mut [Value<ReproducibleEngine<E>>],
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
pub struct ReproducibleGlobal<E: WasmEngine>(E::Global);

impl<E: WasmEngine> WasmGlobal<ReproducibleEngine<E>> for ReproducibleGlobal<E> {
    fn new(
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        value: Value<ReproducibleEngine<E>>,
        mutable: bool,
    ) -> Self {
        Self(<E::Global as WasmGlobal<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            into_value(value),
            mutable,
        ))
    }

    fn ty(&self, ctx: impl AsContext<ReproducibleEngine<E>>) -> GlobalType {
        WasmGlobal::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        new_value: Value<ReproducibleEngine<E>>,
    ) -> anyhow::Result<()> {
        WasmGlobal::set(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            into_value(new_value),
        )
    }

    fn get(
        &self,
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
    ) -> Value<ReproducibleEngine<E>> {
        from_value(WasmGlobal::get(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
        ))
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ReproducibleInstance<E: WasmEngine>(E::Instance);

impl<E: WasmEngine> WasmInstance<ReproducibleEngine<E>> for ReproducibleInstance<E> {
    fn new(
        mut store: impl AsContextMut<ReproducibleEngine<E>>,
        module: &ReproducibleModule<E>,
        imports: &Imports<ReproducibleEngine<E>>,
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
        store: impl AsContext<ReproducibleEngine<E>>,
    ) -> Box<dyn Iterator<Item = Export<ReproducibleEngine<E>>>> {
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
        store: impl AsContext<ReproducibleEngine<E>>,
        name: &str,
    ) -> Option<Extern<ReproducibleEngine<E>>> {
        WasmInstance::get_export(&self.0, store.as_context().as_inner_context(), name)
            .map(from_extern)
    }
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ReproducibleMemory<E: WasmEngine>(E::Memory);

impl<E: WasmEngine> WasmMemory<ReproducibleEngine<E>> for ReproducibleMemory<E> {
    fn new(
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        ty: MemoryType,
    ) -> anyhow::Result<Self> {
        Ok(Self(<E::Memory as WasmMemory<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            ty,
        )?))
    }

    fn ty(&self, ctx: impl AsContext<ReproducibleEngine<E>>) -> MemoryType {
        WasmMemory::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        additional: u32,
    ) -> anyhow::Result<u32> {
        WasmMemory::grow(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            additional,
        )
    }

    fn current_pages(&self, ctx: impl AsContext<ReproducibleEngine<E>>) -> u32 {
        WasmMemory::current_pages(&self.0, ctx.as_context().as_inner_context())
    }

    fn read(
        &self,
        ctx: impl AsContext<ReproducibleEngine<E>>,
        offset: usize,
        buffer: &mut [u8],
    ) -> anyhow::Result<()> {
        WasmMemory::read(&self.0, ctx.as_context().as_inner_context(), offset, buffer)
    }

    fn write(
        &self,
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
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
pub struct ReproducibleModule<E: WasmEngine>(E::Module);

impl<E: WasmEngine> WasmModule<ReproducibleEngine<E>> for ReproducibleModule<E> {
    fn new(engine: &ReproducibleEngine<E>, mut stream: impl std::io::Read) -> anyhow::Result<Self> {
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
    instruction_counter: Option<ReproducibleGlobal<E>>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct ReproducibleStore<T, E: WasmEngine>(E::Store<StoreData<T, E>>);

impl<T, E: WasmEngine> WasmStore<T, ReproducibleEngine<E>> for ReproducibleStore<T, E> {
    fn new(engine: &ReproducibleEngine<E>, data: T) -> Self {
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

    fn engine(&self) -> &ReproducibleEngine<E> {
        ReproducibleEngine::from_ref(WasmStore::engine(&self.0))
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

impl<T, E: WasmEngine> AsContext<ReproducibleEngine<E>> for ReproducibleStore<T, E> {
    type UserState = T;

    fn as_context(&self) -> ReproducibleStoreContext<'_, Self::UserState, E> {
        ReproducibleStoreContext(AsContext::as_context(&self.0))
    }
}

impl<T, E: WasmEngine> AsContextMut<ReproducibleEngine<E>> for ReproducibleStore<T, E> {
    fn as_context_mut(&mut self) -> ReproducibleStoreContextMut<'_, Self::UserState, E> {
        ReproducibleStoreContextMut(AsContextMut::as_context_mut(&mut self.0))
    }
}

#[repr(transparent)]
pub struct ReproducibleStoreContext<'a, T: 'a, E: WasmEngine>(E::StoreContext<'a, StoreData<T, E>>);

impl<'a, T: 'a, E: WasmEngine> WasmStoreContext<'a, T, ReproducibleEngine<E>>
    for ReproducibleStoreContext<'a, T, E>
{
    fn engine(&self) -> &ReproducibleEngine<E> {
        ReproducibleEngine::from_ref(WasmStoreContext::engine(&self.0))
    }

    fn data(&self) -> &T {
        &WasmStoreContext::data(&self.0).data
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContext<ReproducibleEngine<E>>
    for ReproducibleStoreContext<'a, T, E>
{
    type UserState = T;

    fn as_context(&self) -> ReproducibleStoreContext<'_, Self::UserState, E> {
        ReproducibleStoreContext(AsContext::as_context(&self.0))
    }
}

impl<'a, T: 'a, E: WasmEngine> ReproducibleStoreContext<'a, T, E> {
    fn as_inner_context(&self) -> E::StoreContext<'_, StoreData<T, E>> {
        self.0.as_context()
    }
}

#[repr(transparent)]
pub struct ReproducibleStoreContextMut<'a, T: 'a, E: WasmEngine>(
    E::StoreContextMut<'a, StoreData<T, E>>,
);

impl<'a, T: 'a, E: WasmEngine> WasmStoreContext<'a, T, ReproducibleEngine<E>>
    for ReproducibleStoreContextMut<'a, T, E>
{
    fn engine(&self) -> &ReproducibleEngine<E> {
        ReproducibleEngine::from_ref(WasmStoreContext::engine(&self.0))
    }

    fn data(&self) -> &T {
        &WasmStoreContext::data(&self.0).data
    }
}

impl<'a, T: 'a, E: WasmEngine> WasmStoreContextMut<'a, T, ReproducibleEngine<E>>
    for ReproducibleStoreContextMut<'a, T, E>
{
    fn data_mut(&mut self) -> &mut T {
        &mut WasmStoreContextMut::data_mut(&mut self.0).data
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContext<ReproducibleEngine<E>>
    for ReproducibleStoreContextMut<'a, T, E>
{
    type UserState = T;

    fn as_context(&self) -> ReproducibleStoreContext<'_, Self::UserState, E> {
        ReproducibleStoreContext(AsContext::as_context(&self.0))
    }
}

impl<'a, T: 'a, E: WasmEngine> AsContextMut<ReproducibleEngine<E>>
    for ReproducibleStoreContextMut<'a, T, E>
{
    fn as_context_mut(&mut self) -> ReproducibleStoreContextMut<'_, Self::UserState, E> {
        ReproducibleStoreContextMut(AsContextMut::as_context_mut(&mut self.0))
    }
}

impl<'a, T: 'a, E: WasmEngine> ReproducibleStoreContextMut<'a, T, E> {
    fn as_inner_context_mut(&mut self) -> E::StoreContextMut<'_, StoreData<T, E>> {
        self.0.as_context_mut()
    }

    fn get_instruction_counter_global(&mut self) -> &ReproducibleGlobal<E> {
        let mut this = self;

        // NLL cannot prove this to be safe, but Polonius can
        polonius_the_crab::polonius!(|this| -> &'polonius ReproducibleGlobal<E> {
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
pub struct ReproducibleTable<E: WasmEngine>(E::Table);

impl<E: WasmEngine> WasmTable<ReproducibleEngine<E>> for ReproducibleTable<E> {
    fn new(
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        ty: TableType,
        init: Value<ReproducibleEngine<E>>,
    ) -> anyhow::Result<Self> {
        Ok(Self(<E::Table as WasmTable<E>>::new(
            ctx.as_context_mut().as_inner_context_mut(),
            ty,
            into_value(init),
        )?))
    }

    fn ty(&self, ctx: impl AsContext<ReproducibleEngine<E>>) -> TableType {
        WasmTable::ty(&self.0, ctx.as_context().as_inner_context())
    }

    fn size(&self, ctx: impl AsContext<ReproducibleEngine<E>>) -> u32 {
        WasmTable::size(&self.0, ctx.as_context().as_inner_context())
    }

    fn grow(
        &self,
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        delta: u32,
        init: Value<ReproducibleEngine<E>>,
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
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        index: u32,
    ) -> Option<Value<ReproducibleEngine<E>>> {
        WasmTable::get(&self.0, ctx.as_context_mut().as_inner_context_mut(), index).map(from_value)
    }

    fn set(
        &self,
        mut ctx: impl AsContextMut<ReproducibleEngine<E>>,
        index: u32,
        value: Value<ReproducibleEngine<E>>,
    ) -> anyhow::Result<()> {
        WasmTable::set(
            &self.0,
            ctx.as_context_mut().as_inner_context_mut(),
            index,
            into_value(value),
        )
    }
}

const fn as_values<E: WasmEngine>(values: &[Value<ReproducibleEngine<E>>]) -> &[Value<E>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), values.len())
    }
}

fn as_values_mut<E: WasmEngine>(values: &mut [Value<ReproducibleEngine<E>>]) -> &mut [Value<E>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len())
    }
}

const fn from_values<E: WasmEngine>(values: &[Value<E>]) -> &[Value<ReproducibleEngine<E>>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), values.len())
    }
}

fn from_values_mut<E: WasmEngine>(values: &mut [Value<E>]) -> &mut [Value<ReproducibleEngine<E>>] {
    // Safety: all of our WASM runtime type wrappers are transparent newtypes
    #[expect(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len())
    }
}

fn into_value<E: WasmEngine>(value: Value<ReproducibleEngine<E>>) -> Value<E> {
    match value {
        Value::I32(v) => Value::I32(v),
        Value::I64(v) => Value::I64(v),
        Value::F32(v) => Value::F32(v),
        Value::F64(v) => Value::F64(v),
        Value::FuncRef(v) => Value::FuncRef(v.map(|v| v.0)),
        Value::ExternRef(v) => Value::ExternRef(v.map(|v| v.0)),
    }
}

fn from_value<E: WasmEngine>(value: Value<E>) -> Value<ReproducibleEngine<E>> {
    match value {
        Value::I32(v) => Value::I32(v),
        Value::I64(v) => Value::I64(v),
        Value::F32(v) => Value::F32(v),
        Value::F64(v) => Value::F64(v),
        Value::FuncRef(v) => Value::FuncRef(v.map(ReproducibleFunc)),
        Value::ExternRef(v) => Value::ExternRef(v.map(ReproducibleExternRef)),
    }
}

fn into_extern<E: WasmEngine>(value: Extern<ReproducibleEngine<E>>) -> Extern<E> {
    match value {
        Extern::Global(v) => Extern::Global(v.0),
        Extern::Table(v) => Extern::Table(v.0),
        Extern::Memory(v) => Extern::Memory(v.0),
        Extern::Func(v) => Extern::Func(v.0),
    }
}

fn from_extern<E: WasmEngine>(value: Extern<E>) -> Extern<ReproducibleEngine<E>> {
    match value {
        Extern::Global(v) => Extern::Global(ReproducibleGlobal(v)),
        Extern::Table(v) => Extern::Table(ReproducibleTable(v)),
        Extern::Memory(v) => Extern::Memory(ReproducibleMemory(v)),
        Extern::Func(v) => Extern::Func(ReproducibleFunc(v)),
    }
}
