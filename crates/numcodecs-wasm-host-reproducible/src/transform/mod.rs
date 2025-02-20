use std::sync::OnceLock;

use anyhow::{anyhow, Context, Error};
use instcnt::PerfWitInterfaces;
use numcodecs_wasm_host::NumcodecsWitInterfaces;

use crate::{logging::WasiLoggingInterface, stdio::WasiSandboxedStdioInterface};

pub mod instcnt;
pub mod nan;

#[expect(clippy::too_many_lines)] // FIXME
pub fn transform_wasm_component(wasm_component: impl Into<Vec<u8>>) -> Result<Vec<u8>, Error> {
    let NumcodecsWitInterfaces {
        codec: codec_interface,
        ..
    } = NumcodecsWitInterfaces::get();

    // create a new WAC composition graph with the WASI component packages
    //  pre-registered and the numcodecs:wasm/perf interface pre-exported
    let PreparedCompositionGraph {
        graph: wac,
        wasi: wasi_component_packages,
    } = get_prepared_composition_graph()?;
    let mut wac = wac.clone();

    // parse and instantiate the root package, which exports numcodecs:abc/codec
    let numcodecs_codec_package = wac_graph::types::Package::from_bytes(
        &format!("{}", codec_interface.package().name()),
        codec_interface.package().version(),
        wasm_component,
        wac.types_mut(),
    )?;

    let numcodecs_codec_world = &wac.types()[numcodecs_codec_package.ty()];
    let numcodecs_codec_imports = extract_component_ports(&numcodecs_codec_world.imports)?;

    let numcodecs_codec_package = wac.register_package(numcodecs_codec_package)?;
    let numcodecs_codec_instance = wac.instantiate(numcodecs_codec_package);

    // list the imports that the linker will provide
    let linker_provided_imports = [
        &WasiSandboxedStdioInterface::get().stdio,
        &WasiLoggingInterface::get().logging,
    ];

    // initialise the unresolved imports to the imports of the root package
    let mut unresolved_imports = vecmap::VecMap::new();
    for import in &numcodecs_codec_imports {
        unresolved_imports
            .entry(import.clone())
            .or_insert_with(Vec::new)
            .push(numcodecs_codec_instance);
    }

    // track all non-root instances, which may fulfil imports
    let mut package_instances = vecmap::VecMap::new();

    // initialise the queue of required, still to instantiate packages
    //  to the imports of the root package
    let mut required_packages_queue = numcodecs_codec_imports
        .iter()
        .map(|import| import.package().clone())
        .collect::<std::collections::VecDeque<_>>();

    // iterate while not all required packages have been instantiated
    while let Some(required_package) = required_packages_queue.pop_front() {
        if package_instances.contains_key(&required_package) {
            continue;
        }

        // some packages do not need to be instantiated since they will be
        //  provided by the linker
        if linker_provided_imports
            .iter()
            .any(|interface| interface.package() == &required_package)
        {
            continue;
        }

        // find the WASI component package that can fulfil the required package
        let Some(component_package) = wasi_component_packages.iter().find(|component_package| {
            component_package
                .exports
                .iter()
                .any(|export| export.package() == &required_package)
        }) else {
            return Err(anyhow!(
                "WASM component requires unresolved import(s) from package {required_package}"
            ));
        };

        let PackageWithPorts {
            package: component_package,
            imports: component_imports,
            exports: component_exports,
        } = component_package;

        // instantiate the component package
        let component_instance = wac.instantiate(*component_package);

        // try to resolve all imports of the component package ...
        for import in component_imports {
            if let Some(dependency_instance) = package_instances.get(import.package()).copied() {
                // ... if the dependency has already been instantiated,
                //     import its export directly
                let import_str = &format!("{import}");
                let dependency_export =
                    wac.alias_instance_export(dependency_instance, import_str)?;
                wac.set_instantiation_argument(component_instance, import_str, dependency_export)?;
            } else {
                // ... otherwise require the dependency package and store the
                //     import so that it can be resolved later
                required_packages_queue.push_back(import.package().clone());
                unresolved_imports
                    .entry(import.clone())
                    .or_insert_with(Vec::new)
                    .push(component_instance);
            }
        }

        for export in component_exports {
            // register this instance's package so that its exports can later
            //  fulfil more imports
            package_instances.insert(export.package().clone(), component_instance);

            // try to resolve unresolved imports using the export of this package
            if let Some(unresolved_imports) = unresolved_imports.remove(export) {
                let export_str = &format!("{export}");
                let component_export = wac.alias_instance_export(component_instance, export_str)?;
                for import in unresolved_imports {
                    wac.set_instantiation_argument(import, export_str, component_export)?;
                }
            }
        }
    }

    // linker-provided imports will be resolved later
    for provided in linker_provided_imports {
        unresolved_imports.remove(provided);
    }

    if !unresolved_imports.is_empty() {
        return Err(anyhow!(
            "WASM component requires unresolved import(s): {:?}",
            unresolved_imports.into_keys().collect::<Vec<_>>(),
        ));
    }

    // export the numcodecs:abc/codec interface
    let numcodecs_codecs_str = &format!("{codec_interface}");
    let numcodecs_codecs_export =
        wac.alias_instance_export(numcodecs_codec_instance, numcodecs_codecs_str)?;
    wac.export(numcodecs_codecs_export, numcodecs_codecs_str)?;

    // encode the WAC composition graph into a WASM component and validate it
    let wasm = wac.encode(wac_graph::EncodeOptions {
        define_components: true,
        // we do our own validation right below
        validate: false,
        processor: None,
    })?;

    wasmparser::Validator::new_with_features(
        wasmparser::WasmFeaturesInflated {
            // MUST: float operations are required
            //       (and our engine's transformations makes them deterministic)
            floats: true,
            // MUST: codecs and reproducible WASI are implemented as components
            component_model: true,
            // OK: using linear values in component init is deterministic, as
            //     long as the values provided are deterministic
            component_model_values: true,
            // OK: nested component names do not interact with determinism
            component_model_nested_names: true,
            ..crate::engine::DETERMINISTIC_WASM_MODULE_FEATURES
        }
        .into(),
    )
    .validate_all(&wasm)?;

    Ok(wasm)
}

struct PreparedCompositionGraph {
    graph: wac_graph::CompositionGraph,
    wasi: Box<[PackageWithPorts]>,
}

fn get_prepared_composition_graph() -> Result<&'static PreparedCompositionGraph, Error> {
    static PREPARED_COMPOSITION_GRAPH: OnceLock<Result<PreparedCompositionGraph, Error>> =
        OnceLock::new();

    let prepared_composition_graph = PREPARED_COMPOSITION_GRAPH.get_or_init(|| {
        let PerfWitInterfaces {
            perf: perf_interface,
            ..
        } = PerfWitInterfaces::get();

        // create a new WAC composition graph
        let mut wac = wac_graph::CompositionGraph::new();

        // parse and register the WASI component packages
        let wasi_component_packages =
            register_wasi_component_packages(&mut wac)?.into_boxed_slice();

        // create, register, and instantiate the numcodecs:wasm package
        let numcodecs_wasm_perf_instance = instantiate_numcodecs_wasm_perf_package(&mut wac)?;

        // export the numcodecs:wasm/perf interface
        let numcodecs_wasm_perf_str = &format!("{perf_interface}");
        let numcodecs_wasm_perf_export =
            wac.alias_instance_export(numcodecs_wasm_perf_instance, numcodecs_wasm_perf_str)?;
        wac.export(numcodecs_wasm_perf_export, numcodecs_wasm_perf_str)?;

        Ok(PreparedCompositionGraph {
            graph: wac,
            wasi: wasi_component_packages,
        })
    });

    match prepared_composition_graph {
        Ok(prepared_composition_graph) => Ok(prepared_composition_graph),
        Err(err) => Err(anyhow!(err)),
    }
}

struct PackageWithPorts {
    package: wac_graph::PackageId,
    imports: Box<[wasm_component_layer::InterfaceIdentifier]>,
    exports: Box<[wasm_component_layer::InterfaceIdentifier]>,
}

fn register_wasi_component_packages(
    wac: &mut wac_graph::CompositionGraph,
) -> Result<Vec<PackageWithPorts>, Error> {
    let wasi_component_packages = wasi_sandboxed_component_provider::ALL_COMPONENTS
        .iter()
        .map(|(component_name, component_bytes)| -> Result<_, Error> {
            let component_package = wac_graph::types::Package::from_bytes(
                component_name,
                None,
                Vec::from(*component_bytes),
                wac.types_mut(),
            )?;

            let component_world = &wac.types()[component_package.ty()];

            let component_imports = extract_component_ports(&component_world.imports)?;
            let component_exports = extract_component_ports(&component_world.exports)?;

            let component_package = wac.register_package(component_package)?;

            Ok(PackageWithPorts {
                package: component_package,
                imports: component_imports.into_boxed_slice(),
                exports: component_exports.into_boxed_slice(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(wasi_component_packages)
}

fn extract_component_ports(
    ports: &indexmap::IndexMap<String, wac_graph::types::ItemKind>,
) -> Result<Vec<wasm_component_layer::InterfaceIdentifier>, anyhow::Error> {
    ports
        .iter()
        .filter_map(|(import, kind)| match kind {
            wac_graph::types::ItemKind::Instance(_) => Some(
                wasm_component_layer::InterfaceIdentifier::try_from(import.as_str()),
            ),
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()
}

fn instantiate_numcodecs_wasm_perf_package(
    wac: &mut wac_graph::CompositionGraph,
) -> Result<wac_graph::NodeId, Error> {
    let PerfWitInterfaces {
        perf: perf_interface,
        ..
    } = PerfWitInterfaces::get();

    // create, register, and instantiate the numcodecs:wasm/perf package
    let numcodecs_wasm_perf_package = wac_graph::types::Package::from_bytes(
        &format!("{}", perf_interface.package().name()),
        perf_interface.package().version(),
        create_numcodecs_wasm_perf_component()?,
        wac.types_mut(),
    )?;

    let numcodecs_wasm_perf_package = wac.register_package(numcodecs_wasm_perf_package)?;
    let numcodecs_wasm_perf_instance = wac.instantiate(numcodecs_wasm_perf_package);

    Ok(numcodecs_wasm_perf_instance)
}

fn create_numcodecs_wasm_perf_component() -> Result<Vec<u8>, Error> {
    const ROOT: &str = "root";

    let PerfWitInterfaces {
        perf: perf_interface,
        instruction_counter,
    } = PerfWitInterfaces::get();

    let mut module = create_numcodecs_wasm_perf_module();

    let mut resolve = wit_parser::Resolve::new();

    let interface = resolve.interfaces.alloc(wit_parser::Interface {
        name: Some(String::from(perf_interface.name())),
        types: indexmap::IndexMap::new(),
        #[expect(clippy::iter_on_single_items)]
        functions: [(
            String::from(instruction_counter),
            wit_parser::Function {
                name: String::from(instruction_counter),
                kind: wit_parser::FunctionKind::Freestanding,
                params: Vec::new(),
                result: Some(wit_parser::Type::U64),
                docs: wit_parser::Docs { contents: None },
                stability: wit_parser::Stability::Unknown,
            },
        )]
        .into_iter()
        .collect(),
        docs: wit_parser::Docs { contents: None },
        package: None, // The package is linked up below
        stability: wit_parser::Stability::Unknown,
    });

    let package_name = wit_parser::PackageName {
        namespace: String::from(perf_interface.package().name().namespace()),
        name: String::from(perf_interface.package().name().name()),
        version: perf_interface.package().version().cloned(),
    };
    let package = resolve.packages.alloc(wit_parser::Package {
        name: package_name.clone(),
        docs: wit_parser::Docs { contents: None },
        #[expect(clippy::iter_on_single_items)]
        interfaces: [(String::from(perf_interface.name()), interface)]
            .into_iter()
            .collect(),
        worlds: indexmap::IndexMap::new(),
    });
    resolve.package_names.insert(package_name, package);

    if let Some(interface) = resolve.interfaces.get_mut(interface) {
        interface.package = Some(package);
    }

    let world = resolve.worlds.alloc(wit_parser::World {
        name: String::from(ROOT),
        imports: indexmap::IndexMap::new(),
        #[expect(clippy::iter_on_single_items)]
        exports: [(
            wit_parser::WorldKey::Interface(interface),
            wit_parser::WorldItem::Interface {
                id: interface,
                stability: wit_parser::Stability::Unknown,
            },
        )]
        .into_iter()
        .collect(),
        package: None, // The package is linked up below
        docs: wit_parser::Docs { contents: None },
        includes: Vec::new(),
        include_names: Vec::new(),
        stability: wit_parser::Stability::Unknown,
    });

    let root_name = wit_parser::PackageName {
        namespace: String::from(ROOT),
        name: String::from("component"),
        version: perf_interface.package().version().cloned(),
    };
    let root = resolve.packages.alloc(wit_parser::Package {
        name: root_name.clone(),
        docs: wit_parser::Docs { contents: None },
        interfaces: indexmap::IndexMap::new(),
        #[expect(clippy::iter_on_single_items)]
        worlds: [(String::from(ROOT), world)].into_iter().collect(),
    });
    resolve.package_names.insert(root_name, root);

    if let Some(world) = resolve.worlds.get_mut(world) {
        world.package = Some(root);
    }

    wit_component::embed_component_metadata(
        &mut module,
        &resolve,
        world,
        wit_component::StringEncoding::UTF8,
    )?;

    let mut encoder = wit_component::ComponentEncoder::default()
        .module(&module)
        .context("wit_component::ComponentEncoder::module failed")?;

    let component = encoder
        .encode()
        .context("wit_component::ComponentEncoder::encode failed")?;

    Ok(component)
}

fn create_numcodecs_wasm_perf_module() -> Vec<u8> {
    let PerfWitInterfaces {
        perf: perf_interface,
        instruction_counter,
    } = PerfWitInterfaces::get();

    let mut module = wasm_encoder::Module::new();

    // Encode the type section with
    //  types[0] = () -> i64
    let mut types = wasm_encoder::TypeSection::new();
    let ty0 = types.len();
    types.ty().function([], [wasm_encoder::ValType::I64]);
    module.section(&types);

    // Encode the function section with
    //  functions[0] = fn() -> i64 [ types[0] ]
    let mut functions = wasm_encoder::FunctionSection::new();
    let fn0 = functions.len();
    functions.function(ty0);
    module.section(&functions);

    // Encode the export section with
    //  {perf_interface}#{instruction_counter} = functions[0]
    let mut exports = wasm_encoder::ExportSection::new();
    exports.export(
        &format!("{perf_interface}#{instruction_counter}"),
        wasm_encoder::ExportKind::Func,
        fn0,
    );
    module.section(&exports);

    // Encode the code section.
    let mut codes = wasm_encoder::CodeSection::new();
    let mut fn0 = wasm_encoder::Function::new([]);
    fn0.instruction(&wasm_encoder::Instruction::Unreachable);
    fn0.instruction(&wasm_encoder::Instruction::End);
    codes.function(&fn0);
    module.section(&codes);

    // Extract the encoded WASM bytes for this module
    module.finish()
}
