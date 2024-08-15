//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-wasm-guest
//! [crates.io]: https://crates.io/crates/numcodecs-wasm-guest
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-wasm-guest
//! [docs.rs]: https://docs.rs/numcodecs-wasm-guest/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_wasm_guest
//!
//! wasm32-wasi guest-side bindings for the [`numcodecs`] API, which allows you
//! to export one [`StaticCodec`] from a WASM component.

// Required in docs and the [`export_codec`] macro
#[doc(hidden)]
pub use numcodecs;

#[cfg(doc)]
use numcodecs::StaticCodec;

#[cfg(target_arch = "wasm32")]
use ::{
    numcodecs::{Codec, StaticCodec},
    serde::{Serialize, Serializer},
};

#[cfg(target_arch = "wasm32")]
mod convert;

#[cfg(target_arch = "wasm32")]
use crate::{
    bindings::exports::numcodecs::abc::codec as wit,
    convert::{from_wit_any_array, into_wit_any_array, into_wit_error},
};

#[doc(hidden)]
#[allow(clippy::missing_safety_doc)]
pub mod bindings {
    wit_bindgen::generate!({
        world: "numcodecs:abc/exports@0.1.0",
        with: {
            "numcodecs:abc/codec@0.1.0": generate,
        },
        pub_export_macro: true,
    });
}

#[macro_export]
/// Export a [`StaticCodec`] type using the WASM component model.
///
/// ```rust,ignore
/// # use numcodecs_wasm_guest::export_codec;
///
/// struct MyCodec;
///
/// impl numcodecs::Codec for MyCodec {
///     // ...
/// }
///
/// impl numcodecs::StaticCodec for MyCodec {
///     // ...
/// }
///
/// export_codec!(MyCodec);
/// ```
macro_rules! export_codec {
    ($codec:ident) => {
        const _: () = {
            const fn can_only_export_static_codec<T: $crate::numcodecs::StaticCodec>() {}

            can_only_export_static_codec::<$codec>()
        };

        #[cfg(target_arch = "wasm32")]
        $crate::bindings::export!($codec with_types_in $crate::bindings);
    };
}

#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
impl<T: StaticCodec> wit::Guest for T {
    type Codec = Self;

    fn codec_id() -> String {
        String::from(<Self as StaticCodec>::CODEC_ID)
    }
}

#[cfg(target_arch = "wasm32")]
impl<T: StaticCodec> wit::GuestCodec for T {
    fn from_config(config: String) -> Result<wit::Codec, wit::Error> {
        match <T as StaticCodec>::from_config(&mut serde_json::Deserializer::from_str(&config)) {
            Ok(codec) => Ok(wit::Codec::new(codec)),
            Err(err) => {
                let err = format_serde_error::SerdeError::new(config, err);
                Err(into_wit_error(err))
            }
        }
    }

    fn encode(&self, data: wit::AnyArray) -> Result<wit::AnyArray, wit::Error> {
        let data = match from_wit_any_array(data) {
            Ok(data) => data,
            Err(err) => return Err(into_wit_error(err)),
        };

        match <T as Codec>::encode(self, data.into_cow()) {
            Ok(encoded) => match into_wit_any_array(encoded) {
                Ok(encoded) => Ok(encoded),
                Err(err) => Err(into_wit_error(err)),
            },
            Err(err) => Err(into_wit_error(err)),
        }
    }

    fn decode(&self, encoded: wit::AnyArray) -> Result<wit::AnyArray, wit::Error> {
        let encoded = match from_wit_any_array(encoded) {
            Ok(encoded) => encoded,
            Err(err) => return Err(into_wit_error(err)),
        };

        match <T as Codec>::decode(self, encoded.into_cow()) {
            Ok(decoded) => match into_wit_any_array(decoded) {
                Ok(decoded) => Ok(decoded),
                Err(err) => Err(into_wit_error(err)),
            },
            Err(err) => Err(into_wit_error(err)),
        }
    }

    fn get_config(&self) -> Result<String, wit::Error> {
        struct SerializeWrapper<'a, T: Codec>(&'a T);
        impl<'a, T: Codec> Serialize for SerializeWrapper<'a, T> {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                <T as Codec>::get_config(self.0, serializer)
            }
        }

        match serde_json::to_string(&SerializeWrapper(self)) {
            Ok(config) => Ok(config),
            Err(err) => Err(into_wit_error(err)),
        }
    }
}
