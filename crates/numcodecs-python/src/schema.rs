use std::{
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
};

use pyo3::{intern, prelude::*, sync::GILOnceCell};
use pythonize::{depythonize, PythonizeError};
use schemars::Schema;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::{export::RustCodec, PyCodecClass};

macro_rules! once {
    ($py:ident, $module:literal $(, $path:literal)*) => {{
        fn once(py: Python) -> Result<&Bound<PyAny>, PyErr> {
            static ONCE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
            Ok(ONCE.get_or_try_init(py, || -> Result<Py<PyAny>, PyErr> {
                Ok(py
                    .import(intern!(py, $module))?
                    $(.getattr(intern!(py, $path))?)*
                    .unbind())
            })?.bind(py))
        }

        once($py)
    }};
}

pub fn schema_from_codec_class(
    py: Python,
    class: &Bound<PyCodecClass>,
) -> Result<Schema, SchemaError> {
    if let Ok(schema) = class.getattr(intern!(py, RustCodec::SCHEMA_ATTRIBUTE)) {
        return depythonize(&schema)
            .map_err(|err| SchemaError::InvalidCachedJsonSchema { source: err });
    }

    let mut schema = Schema::default();

    {
        let schema = schema.ensure_object();

        schema.insert(String::from("type"), Value::String(String::from("object")));

        if let Ok(init) = class.getattr(intern!(py, "__init__")) {
            let mut properties = Map::new();
            let mut additional_properties = false;
            let mut required = Vec::new();

            let object_init = once!(py, "builtins", "object", "__init__")?;
            let signature = once!(py, "inspect", "signature")?;
            let empty_parameter = once!(py, "inspect", "Parameter", "empty")?;
            let args_parameter = once!(py, "inspect", "Parameter", "VAR_POSITIONAL")?;
            let kwargs_parameter = once!(py, "inspect", "Parameter", "VAR_KEYWORD")?;

            for (i, param) in signature
                .call1((&init,))?
                .getattr(intern!(py, "parameters"))?
                .call_method0(intern!(py, "items"))?
                .try_iter()?
                .enumerate()
            {
                let (name, param): (String, Bound<PyAny>) = param?.extract()?;

                if i == 0 && name == "self" {
                    continue;
                }

                let kind = param.getattr(intern!(py, "kind"))?;

                if kind.eq(args_parameter)? && !init.eq(object_init)? {
                    return Err(SchemaError::ArgsParameterInSignature);
                }

                if kind.eq(kwargs_parameter)? {
                    additional_properties = true;
                } else {
                    let default = param.getattr(intern!(py, "default"))?;

                    let mut parameter = Map::new();

                    if default.eq(empty_parameter)? {
                        required.push(Value::String(name.clone()));
                    } else {
                        let default = depythonize(&default).map_err(|err| {
                            SchemaError::InvalidParameterDefault {
                                name: name.clone(),
                                source: err,
                            }
                        })?;
                        parameter.insert(String::from("default"), default);
                    }

                    properties.insert(name, Value::Object(parameter));
                }
            }

            schema.insert(
                String::from("additionalProperties"),
                Value::Bool(additional_properties),
            );
            schema.insert(String::from("properties"), Value::Object(properties));
            schema.insert(String::from("required"), Value::Array(required));
        } else {
            schema.insert(String::from("additionalProperties"), Value::Bool(true));
        }

        if let Ok(doc) = class.getattr(intern!(py, "__doc__")) {
            if !doc.is_none() {
                let doc: String = doc
                    .extract()
                    .map_err(|err| SchemaError::InvalidClassDocs { source: err })?;
                schema.insert(String::from("description"), Value::String(doc));
            }
        }

        let name = class
            .getattr(intern!(py, "__name__"))
            .and_then(|name| name.extract())
            .map_err(|err| SchemaError::InvalidClassName { source: err })?;
        schema.insert(String::from("title"), Value::String(name));

        schema.insert(
            String::from("$schema"),
            Value::String(String::from("https://json-schema.org/draft/2020-12/schema")),
        );
    }

    Ok(schema)
}

pub fn docs_from_schema(schema: &Schema) -> Option<String> {
    let parameters = parameters_from_schema(schema);
    let schema = schema.as_object()?;

    let mut docs = String::new();

    if let Some(Value::String(description)) = schema.get("description") {
        docs.push_str(&derust_doc_comment(description));
        docs.push_str("\n\n");
    }

    if !parameters.named.is_empty() || parameters.additional {
        docs.push_str("Parameters\n----------\n");
    }

    for parameter in &parameters.named {
        docs.push_str(parameter.name);

        docs.push_str(" : ...");

        if !parameter.required {
            docs.push_str(", optional");
        }

        #[expect(clippy::format_push_string)] // FIXME
        if let Some(default) = parameter.default {
            docs.push_str(", default = ");
            docs.push_str(&format!("{default}"));
        }

        docs.push('\n');

        if let Some(info) = &parameter.docs {
            docs.push_str("    ");
            docs.push_str(&info.replace('\n', "\n    "));
        }

        docs.push('\n');
    }

    if parameters.additional {
        docs.push_str("**kwargs\n");
        docs.push_str("    ");

        if parameters.named.is_empty() {
            docs.push_str("This codec takes *any* parameters.");
        } else {
            docs.push_str("This codec takes *any* additional parameters.");
        }
    } else if parameters.named.is_empty() {
        docs.push_str("This codec does *not* take any parameters.");
    }

    docs.truncate(docs.trim_end().len());

    Some(docs)
}

pub fn signature_from_schema(schema: &Schema) -> String {
    let parameters = parameters_from_schema(schema);

    let mut signature = String::new();
    signature.push_str("self");

    for parameter in parameters.named {
        signature.push_str(", ");
        signature.push_str(parameter.name);

        #[expect(clippy::format_push_string)] // FIXME
        if let Some(default) = parameter.default {
            signature.push('=');
            signature.push_str(&format!("{default}"));
        } else if !parameter.required {
            signature.push_str("=None");
        }
    }

    if parameters.additional {
        signature.push_str(", **kwargs");
    }

    signature
}

fn parameters_from_schema(schema: &Schema) -> Parameters {
    // schema = true means that any parameters are allowed
    if schema.as_bool() == Some(true) {
        return Parameters {
            named: Vec::new(),
            additional: true,
        };
    }

    // schema = false means that no config is valid
    // we approximate that by saying that no parameters are allowed
    let Some(schema) = schema.as_object() else {
        return Parameters {
            named: Vec::new(),
            additional: false,
        };
    };

    let mut parameters = Vec::new();

    let required = match schema.get("required") {
        Some(Value::Array(required)) => &**required,
        _ => &[],
    };

    // extract the top-level parameters
    if let Some(Value::Object(properties)) = schema.get("properties") {
        for (name, parameter) in properties {
            parameters.push(Parameter::new(name, parameter, required));
        }
    }

    let mut additional = false;

    extend_parameters_from_one_of_schema(schema, &mut parameters, &mut additional);

    // iterate over allOf to handle flattened enums
    if let Some(Value::Array(all)) = schema.get("allOf") {
        for variant in all {
            if let Some(variant) = variant.as_object() {
                extend_parameters_from_one_of_schema(variant, &mut parameters, &mut additional);
            }
        }
    }

    // sort parameters by name and so that required parameters come first
    parameters.sort_by_key(|p| (!p.required, p.name));

    additional = match (
        schema.get("additionalProperties"),
        schema.get("unevaluatedProperties"),
    ) {
        (Some(Value::Bool(false)), None) => additional,
        (None | Some(Value::Bool(false)), Some(Value::Bool(false))) => false,
        _ => true,
    };

    Parameters {
        named: parameters,
        additional,
    }
}

fn extend_parameters_from_one_of_schema<'a>(
    schema: &'a Map<String, Value>,
    parameters: &mut Vec<Parameter<'a>>,
    additional: &mut bool,
) {
    // iterate over oneOf to handle top-level or flattened enums
    if let Some(Value::Array(variants)) = schema.get("oneOf") {
        let mut variant_parameters = HashMap::new();

        for (generation, schema) in variants.iter().enumerate() {
            // if any variant allows additional parameters, the top-level also
            //  allows additional parameters
            #[expect(clippy::unnested_or_patterns)]
            if let Some(schema) = schema.as_object() {
                *additional |= !matches!(
                    (
                        schema.get("additionalProperties"),
                        schema.get("unevaluatedProperties")
                    ),
                    (Some(Value::Bool(false)), None)
                        | (None, Some(Value::Bool(false)))
                        | (Some(Value::Bool(false)), Some(Value::Bool(false)))
                );
            }

            let required = match schema.get("required") {
                Some(Value::Array(required)) => &**required,
                _ => &[],
            };
            let variant_docs = match schema.get("description") {
                Some(Value::String(docs)) => Some(derust_doc_comment(docs)),
                _ => None,
            };

            // extract the per-variant parameters and check for tag parameters
            if let Some(Value::Object(properties)) = schema.get("properties") {
                for (name, parameter) in properties {
                    match variant_parameters.entry(name) {
                        Entry::Vacant(entry) => {
                            entry.insert(VariantParameter::new(
                                generation,
                                name,
                                parameter,
                                required,
                                variant_docs.clone(),
                            ));
                        }
                        Entry::Occupied(mut entry) => {
                            entry.get_mut().merge(
                                generation,
                                name,
                                parameter,
                                required,
                                variant_docs.clone(),
                            );
                        }
                    }
                }
            }

            // ensure that only parameters in all variants are required or tags
            for parameter in variant_parameters.values_mut() {
                parameter.update_generation(generation);
            }
        }

        // merge the variant parameters into the top-level parameters
        parameters.extend(
            variant_parameters
                .into_values()
                .map(VariantParameter::into_parameter),
        );
    }
}

fn derust_doc_comment(docs: &str) -> Cow<str> {
    if docs.trim() != docs {
        return Cow::Borrowed(docs);
    }

    if !docs
        .split('\n')
        .skip(1)
        .all(|l| l.trim().is_empty() || l.starts_with(' '))
    {
        return Cow::Borrowed(docs);
    }

    Cow::Owned(docs.replace("\n ", "\n"))
}

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("codec class' cached config schema is invalid")]
    InvalidCachedJsonSchema { source: PythonizeError },
    #[error("extracting the codec signature failed")]
    SignatureExtraction {
        #[from]
        source: PyErr,
    },
    #[error("codec's signature must not contain an `*args` parameter")]
    ArgsParameterInSignature,
    #[error("{name} parameter's default value is invalid")]
    InvalidParameterDefault {
        name: String,
        source: PythonizeError,
    },
    #[error("codec class's `__doc__` must be a string")]
    InvalidClassDocs { source: PyErr },
    #[error("codec class must have a string `__name__`")]
    InvalidClassName { source: PyErr },
}

struct Parameters<'a> {
    named: Vec<Parameter<'a>>,
    additional: bool,
}

struct Parameter<'a> {
    name: &'a str,
    required: bool,
    default: Option<&'a Value>,
    docs: Option<Cow<'a, str>>,
}

impl<'a> Parameter<'a> {
    #[must_use]
    pub fn new(name: &'a str, parameter: &'a Value, required: &[Value]) -> Self {
        Self {
            name,
            required: required
                .iter()
                .any(|r| matches!(r, Value::String(n) if n == name)),
            default: parameter.get("default"),
            docs: match parameter.get("description") {
                Some(Value::String(docs)) => Some(derust_doc_comment(docs)),
                _ => None,
            },
        }
    }
}

struct VariantParameter<'a> {
    generation: usize,
    parameter: Parameter<'a>,
    #[expect(clippy::type_complexity)]
    tag_docs: Option<Vec<(&'a Value, Option<Cow<'a, str>>)>>,
}

impl<'a> VariantParameter<'a> {
    #[must_use]
    pub fn new(
        generation: usize,
        name: &'a str,
        parameter: &'a Value,
        required: &[Value],
        variant_docs: Option<Cow<'a, str>>,
    ) -> Self {
        let r#const = parameter.get("const");

        let mut parameter = Parameter::new(name, parameter, required);
        parameter.required &= generation == 0;

        let tag_docs = match r#const {
            // a tag parameter must be introduced in the first generation
            Some(r#const) if generation == 0 => {
                let docs = parameter.docs.take().or(variant_docs);
                Some(vec![(r#const, docs)])
            }
            _ => None,
        };

        Self {
            generation,
            parameter,
            tag_docs,
        }
    }

    pub fn merge(
        &mut self,
        generation: usize,
        name: &'a str,
        parameter: &'a Value,
        required: &[Value],
        variant_docs: Option<Cow<'a, str>>,
    ) {
        self.generation = generation;

        let r#const = parameter.get("const");

        let parameter = Parameter::new(name, parameter, required);

        self.parameter.required &= parameter.required;
        if self.parameter.default != parameter.default {
            self.parameter.default = None;
        }

        if let Some(tag_docs) = &mut self.tag_docs {
            // we're building docs for a tag-like parameter
            if let Some(r#const) = r#const {
                tag_docs.push((r#const, parameter.docs.or(variant_docs)));
            } else {
                // mixing tag and non-tag parameter => no docs
                self.tag_docs = None;
                self.parameter.docs = None;
            }
        } else {
            // we're building docs for a normal parameter
            if r#const.is_none() {
                // we only accept always matching docs for normal parameters
                if self.parameter.docs != parameter.docs {
                    self.parameter.docs = None;
                }
            } else {
                // mixing tag and non-tag parameter => no docs
                self.tag_docs = None;
            }
        }
    }

    pub fn update_generation(&mut self, generation: usize) {
        if self.generation < generation {
            // required and tag parameters must appear in all generations
            self.parameter.required = false;
            self.tag_docs = None;
        }
    }

    #[must_use]
    pub fn into_parameter(mut self) -> Parameter<'a> {
        if let Some(tag_docs) = self.tag_docs {
            let mut docs = String::new();

            #[expect(clippy::format_push_string)] // FIXME
            for (tag, tag_docs) in tag_docs {
                docs.push_str(" - ");
                docs.push_str(&format!("{tag}"));
                if let Some(tag_docs) = tag_docs {
                    docs.push_str(": ");
                    docs.push_str(&tag_docs.replace('\n', "\n    "));
                }
                docs.push_str("\n\n");
            }

            docs.truncate(docs.trim_end().len());

            self.parameter.docs = Some(Cow::Owned(docs));
        }

        self.parameter
    }
}

#[cfg(test)]
mod tests {
    use schemars::{schema_for, JsonSchema};

    use super::*;

    #[test]
    fn schema() {
        assert_eq!(
            format!("{}", schema_for!(MyCodec).to_value()),
            r#"{"type":"object","properties":{"param":{"type":["integer","null"],"format":"int32","description":"An optional integer value."}},"unevaluatedProperties":false,"oneOf":[{"type":"object","description":"Mode a.\n\n It gets another line.","properties":{"value":{"type":"boolean","description":"A boolean value. And some really, really, really, long first\n line that wraps around.\n\n With multiple lines of comments."},"common":{"type":"string","description":"A common string value.\n\n Something else here."},"mode":{"type":"string","const":"A"}},"required":["mode","value","common"]},{"type":"object","description":"Mode b.","properties":{"common":{"type":"string","description":"A common string value.\n\n Something else here."},"mode":{"type":"string","const":"B"}},"required":["mode","common"]}],"description":"A codec that does something on encoding and decoding.\n\n With multiple lines of comments.","title":"MyCodec","$schema":"https://json-schema.org/draft/2020-12/schema"}"#
        );
    }

    #[test]
    fn docs() {
        assert_eq!(
            docs_from_schema(&schema_for!(MyCodec)).as_deref(),
            Some(
                r#"A codec that does something on encoding and decoding.

With multiple lines of comments.

Parameters
----------
common : ...
    A common string value.
    
    Something else here.
mode : ...
     - "A": Mode a.
        
        It gets another line.
    
     - "B": Mode b.
param : ..., optional
    An optional integer value.
value : ..., optional
    A boolean value. And some really, really, really, long first
    line that wraps around.
    
    With multiple lines of comments."#
            )
        );
    }

    #[test]
    fn signature() {
        assert_eq!(
            signature_from_schema(&schema_for!(MyCodec)),
            "self, common, mode, param=None, value=None",
        );
    }

    #[expect(dead_code)]
    #[derive(JsonSchema)]
    #[schemars(deny_unknown_fields)]
    /// A codec that does something on encoding and decoding.
    ///
    /// With multiple lines of comments.
    struct MyCodec {
        /// An optional integer value.
        #[schemars(default, skip_serializing_if = "Option::is_none")]
        param: Option<i32>,
        /// The flattened configuration.
        #[schemars(flatten)]
        config: Config,
    }

    #[expect(dead_code)]
    #[derive(JsonSchema)]
    #[schemars(tag = "mode")]
    #[schemars(deny_unknown_fields)]
    enum Config {
        /// Mode a.
        ///
        /// It gets another line.
        A {
            /// A boolean value. And some really, really, really, long first
            /// line that wraps around.
            ///
            /// With multiple lines of comments.
            value: bool,
            /// A common string value.
            ///
            /// Something else here.
            common: String,
        },
        /// Mode b.
        B {
            /// A common string value.
            ///
            /// Something else here.
            common: String,
        },
    }
}
