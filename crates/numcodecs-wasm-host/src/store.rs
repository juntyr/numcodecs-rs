use wasm_component_layer::{Func, Instance, ResourceBorrow, ResourceOwn, Store, TypedFunc, Value};
use wasm_runtime_layer::backend::WasmEngine;

pub trait ErasedWasmStore {
    fn call_func(
        &mut self,
        func: &Func,
        arguments: &[Value],
        results: &mut [Value],
    ) -> Result<(), anyhow::Error>;
    fn call_typed_str_func(
        &mut self,
        func: &TypedFunc<(), String>,
    ) -> Result<String, anyhow::Error>;
    fn call_typed_u64_func(&mut self, func: &TypedFunc<(), u64>) -> Result<u64, anyhow::Error>;
    fn borrow_resource(&mut self, resource: &ResourceOwn) -> Result<ResourceBorrow, anyhow::Error>;
    fn drop_resource(&mut self, resource: &ResourceOwn) -> Result<(), anyhow::Error>;
    fn drop_instance(&mut self, instance: &Instance) -> Result<Vec<anyhow::Error>, anyhow::Error>;
}

impl<T, E: WasmEngine> ErasedWasmStore for Store<T, E> {
    fn call_func(
        &mut self,
        func: &Func,
        arguments: &[Value],
        results: &mut [Value],
    ) -> Result<(), anyhow::Error> {
        func.call(self, arguments, results)
    }

    fn call_typed_str_func(
        &mut self,
        func: &TypedFunc<(), String>,
    ) -> Result<String, anyhow::Error> {
        func.call(self, ())
    }

    fn call_typed_u64_func(&mut self, func: &TypedFunc<(), u64>) -> Result<u64, anyhow::Error> {
        func.call(self, ())
    }

    fn borrow_resource(&mut self, resource: &ResourceOwn) -> Result<ResourceBorrow, anyhow::Error> {
        resource.borrow(self)
    }

    fn drop_resource(&mut self, resource: &ResourceOwn) -> Result<(), anyhow::Error> {
        resource.drop(self)
    }

    fn drop_instance(&mut self, instance: &Instance) -> Result<Vec<anyhow::Error>, anyhow::Error> {
        instance.drop(self)
    }
}
