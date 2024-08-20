use pyo3::{intern, prelude::*, sync::GILOnceCell};
use pythonize::{depythonize_bound, PythonizeError};
use schemars::Schema;
use serde_json::{Map, Value};
use thiserror::Error;

use crate::PyCodecClass;

macro_rules! once {
    ($py:ident, $module:literal $(, $path:literal)*) => {{
        fn once(py: Python) -> Result<&Bound<PyAny>, PyErr> {
            static ONCE: GILOnceCell<Py<PyAny>> = GILOnceCell::new();
            Ok(ONCE.get_or_try_init(py, || -> Result<Py<PyAny>, PyErr> {
                Ok(py
                    .import_bound(intern!(py, $module))?
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
    if let Ok(schema) = class.getattr(intern!(py, "__schema__")) {
        return depythonize_bound(schema)
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
                .iter()?
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
                        let default = depythonize_bound(default).map_err(|err| {
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

pub fn docs_from_schema(schema: &Schema, codec_id: &str) -> Option<String> {
    let parameters = parameters_from_schema(schema);
    let schema = schema.as_object()?;

    let mut docs = String::new();

    docs.push_str("# ");
    docs.push_str(codec_id);

    if let Some(Value::String(title)) = schema.get("title") {
        docs.push_str(" (");
        docs.push_str(title);
        docs.push(')');
    }

    docs.push_str("\n\n");

    if let Some(Value::String(description)) = schema.get("description") {
        docs.push_str(description);
        docs.push_str("\n\n");
    }

    docs.push_str("## Parameters\n\n");

    for parameter in &parameters.named {
        docs.push_str(" - ");
        docs.push_str(parameter.name);

        docs.push_str(" (");

        if parameter.required {
            docs.push_str("required");
        } else {
            docs.push_str("optional");
        }

        if let Some(default) = parameter.default {
            docs.push_str(", default = `");
            docs.push_str(&format!("{default}"));
            docs.push('`');
        }

        docs.push(')');

        if let Some(info) = parameter.docs {
            docs.push_str(": ");
            docs.push_str(info);
        }

        docs.push('\n');
    }

    if parameters.named.is_empty() {
        if parameters.additional {
            docs.push_str("This codec takes *any* parameters.");
        } else {
            docs.push_str("This codec does *not* take any parameters.");
        }
    } else if parameters.additional {
        docs.push_str("\nThis codec takes *any* additional parameters.");
    }

    docs.truncate(docs.trim_end().len());

    Some(docs)
}

fn parameters_from_schema(schema: &Schema) -> Parameters {
    if schema.as_bool() == Some(true) {
        return Parameters {
            named: Vec::new(),
            additional: true,
        };
    }

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

    if let Some(Value::Object(properties)) = schema.get("properties") {
        for (name, parameter) in properties {
            parameters.push(Parameter {
                name,
                required: required
                    .iter()
                    .any(|r| matches!(r, Value::String(n) if n == name)),
                default: parameter.get("default"),
                docs: match parameter.get("description") {
                    Some(Value::String(docs)) => Some(docs),
                    _ => None,
                },
            });
        }
    }

    // TODO: handle oneOf recursion

    Parameters {
        named: parameters,
        additional: !matches!(schema.get("additionalProperties"), Some(Value::Bool(false))),
    }
}

#[derive(Debug, Error)]
pub enum SchemaError {
    #[error("codec class' `__schema__` is invalid")]
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
    docs: Option<&'a str>,
}
