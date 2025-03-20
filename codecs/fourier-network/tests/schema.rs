#![expect(missing_docs)]

use numcodecs::{DynCodecType, StaticCodecType};
use numcodecs_fourier_network::FourierNetworkCodec;

use ::{
    burn as _, bytemuck_derive as _, itertools as _, log as _, ndarray as _, num_traits as _,
    schemars as _, serde as _, serde_json as _, simple_logger as _, thiserror as _,
};

// FIXME: see https://github.com/tracel-ai/burn/issues/2876
use ::{bincode as _, bincode_derive as _};

#[test]
fn schema() {
    let schema = format!(
        "{:#}",
        StaticCodecType::<FourierNetworkCodec>::of()
            .codec_config_schema()
            .to_value()
    );

    if schema != include_str!("schema.json") {
        panic!("FourierNetwork schema has changed\n===\n{schema}\n===");
    }
}
