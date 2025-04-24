//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.85.0-blue
//! [repo]: https://github.com/juntyr/numcodecs-rs
//!
//! [Latest Version]: https://img.shields.io/crates/v/numcodecs-fourier-network
//! [crates.io]: https://crates.io/crates/numcodecs-fourier-network
//!
//! [Rust Doc Crate]: https://img.shields.io/docsrs/numcodecs-fourier-network
//! [docs.rs]: https://docs.rs/numcodecs-fourier-network/
//!
//! [Rust Doc Main]: https://img.shields.io/badge/docs-main-blue
//! [docs]: https://juntyr.github.io/numcodecs-rs/numcodecs_fourier_network
//!
//! Fourier feature neural network codec implementation for the [`numcodecs`] API.

#![expect(clippy::multiple_crate_versions)]

use std::{borrow::Cow, num::NonZeroUsize, ops::AddAssign};

use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    module::{Module, Param},
    nn::loss::{MseLoss, Reduction},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    record::{
        BinBytesRecorder, DoublePrecisionSettings, FullPrecisionSettings, PrecisionSettings,
        Record, Recorder, RecorderError,
    },
    tensor::{
        Distribution, Element as BurnElement, Float, Tensor, TensorData, backend::AutodiffBackend,
    },
};
use itertools::Itertools;
use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Data, Dimension, Ix1, Order, Zip};
use num_traits::{ConstOne, ConstZero, Float as FloatTrait, FromPrimitive};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig, StaticCodecVersion,
};
use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[cfg(test)]
use ::serde_json as _;

mod modules;

use modules::{Model, ModelConfig, ModelExtra, ModelRecord};

type FourierNetworkCodecVersion = StaticCodecVersion<0, 1, 0>;

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
/// Fourier network codec which trains and overfits a fourier feature neural
/// network on encoding and predicts during decoding.
///
/// The approach is based on the papers by Tancik et al. 2020
/// (<https://dl.acm.org/doi/abs/10.5555/3495724.3496356>)
/// and by Huang and Hoefler 2020 (<https://arxiv.org/abs/2210.12538>).
pub struct FourierNetworkCodec {
    /// The number of Fourier features that the data coordinates are projected to
    pub fourier_features: NonZeroUsize,
    /// The standard deviation of the Fourier features
    pub fourier_scale: Positive<f64>,
    /// The number of blocks in the network
    pub num_blocks: NonZeroUsize,
    /// The learning rate for the `Adam` optimizer
    pub learning_rate: Positive<f64>,
    /// The number of epochs for which the network is trained
    pub num_epochs: usize,
    /// The optional mini-batch size used during training
    ///
    /// Setting the mini-batch size to `None` disables the use of batching,
    /// i.e. the network is trained using one large batch that includes the
    /// full data.
    #[serde(deserialize_with = "deserialize_required_option")]
    #[schemars(required, extend("type" = ["integer", "null"]))]
    pub mini_batch_size: Option<NonZeroUsize>,
    /// The seed for the random number generator used during encoding
    pub seed: u64,
    /// The codec's encoding format version. Do not provide this parameter explicitly.
    #[serde(default, rename = "_version")]
    pub version: FourierNetworkCodecVersion,
}

// using this wrapper function makes an Option<T> required
fn deserialize_required_option<'de, T: serde::Deserialize<'de>, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

impl Codec for FourierNetworkCodec {
    type Error = FourierNetworkCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                encode::<f32, _, _, Autodiff<NdArray<f32>>>(
                    &NdArrayDevice::Cpu,
                    data,
                    self.fourier_features,
                    self.fourier_scale,
                    self.num_blocks,
                    self.learning_rate,
                    self.num_epochs,
                    self.mini_batch_size,
                    self.seed,
                )?
                .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                encode::<f64, _, _, Autodiff<NdArray<f64>>>(
                    &NdArrayDevice::Cpu,
                    data,
                    self.fourier_features,
                    self.fourier_scale,
                    self.num_blocks,
                    self.learning_rate,
                    self.num_epochs,
                    self.mini_batch_size,
                    self.seed,
                )?
                .into_dyn(),
            )),
            encoded => Err(FourierNetworkCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }

    fn decode(&self, _encoded: AnyCowArray) -> Result<AnyArray, Self::Error> {
        Err(FourierNetworkCodecError::MissingDecodingOutput)
    }

    fn decode_into(
        &self,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), Self::Error> {
        let AnyArrayView::U8(encoded) = encoded else {
            return Err(FourierNetworkCodecError::EncodedDataNotBytes {
                dtype: encoded.dtype(),
            });
        };

        let Ok(encoded): Result<ArrayBase<_, Ix1>, _> = encoded.view().into_dimensionality() else {
            return Err(FourierNetworkCodecError::EncodedDataNotOneDimensional {
                shape: encoded.shape().to_vec(),
            });
        };

        match decoded {
            AnyArrayViewMut::F32(decoded) => decode_into::<f32, _, _, NdArray<f32>>(
                &NdArrayDevice::Cpu,
                encoded,
                decoded,
                self.fourier_features,
                self.num_blocks,
            ),
            AnyArrayViewMut::F64(decoded) => decode_into::<f64, _, _, NdArray<f64>>(
                &NdArrayDevice::Cpu,
                encoded,
                decoded,
                self.fourier_features,
                self.num_blocks,
            ),
            decoded => Err(FourierNetworkCodecError::UnsupportedDtype(decoded.dtype())),
        }
    }
}

impl StaticCodec for FourierNetworkCodec {
    const CODEC_ID: &'static str = "fourier-network.rs";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[expect(clippy::derive_partial_eq_without_eq)] // floats are not Eq
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash)]
/// Positive floating point number
pub struct Positive<T: FloatTrait>(T);

impl Serialize for Positive<f64> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(self.0)
    }
}

impl<'de> Deserialize<'de> for Positive<f64> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let x = f64::deserialize(deserializer)?;

        if x > 0.0 {
            Ok(Self(x))
        } else {
            Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Float(x),
                &"a positive value",
            ))
        }
    }
}

impl JsonSchema for Positive<f64> {
    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("PositiveF64")
    }

    fn schema_id() -> Cow<'static, str> {
        Cow::Borrowed(concat!(module_path!(), "::", "Positive<f64>"))
    }

    fn json_schema(_gen: &mut SchemaGenerator) -> Schema {
        json_schema!({
            "type": "number",
            "exclusiveMinimum": 0.0
        })
    }
}

#[derive(Debug, Error)]
/// Errors that may occur when applying the [`FourierNetworkCodec`].
pub enum FourierNetworkCodecError {
    /// [`FourierNetworkCodec`] does not support the dtype
    #[error("FourierNetwork does not support the dtype {0}")]
    UnsupportedDtype(AnyArrayDType),
    /// [`FourierNetworkCodec`] does not support non-finite (infinite or NaN) floating
    /// point data
    #[error("FourierNetwork does not support non-finite (infinite or NaN) floating point data")]
    NonFiniteData,
    /// [`FourierNetworkCodec`] failed during a neural network computation
    #[error("FourierNetwork failed during a neural network computation")]
    NeuralNetworkError {
        /// The source of the error
        #[from]
        source: NeuralNetworkError,
    },
    /// [`FourierNetworkCodec`] must be provided the output array during decoding
    #[error("FourierNetwork must be provided the output array during decoding")]
    MissingDecodingOutput,
    /// [`FourierNetworkCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different dtype
    #[error(
        "FourierNetwork can only decode one-dimensional byte arrays but received an array of dtype {dtype}"
    )]
    EncodedDataNotBytes {
        /// The unexpected dtype of the encoded array
        dtype: AnyArrayDType,
    },
    /// [`FourierNetworkCodec`] can only decode one-dimensional byte arrays but
    /// received an array of a different shape
    #[error(
        "FourierNetwork can only decode one-dimensional byte arrays but received a byte array of shape {shape:?}"
    )]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`FourierNetworkCodec`] cannot decode into the provided array
    #[error("FourierNetwork cannot decode into the provided array")]
    MismatchedDecodeIntoArray {
        /// The source of the error
        #[from]
        source: AnyArrayAssignError,
    },
}

#[derive(Debug, Error)]
#[error(transparent)]
/// Opaque error for when an error occurs in the neural network
pub struct NeuralNetworkError(RecorderError);

/// Floating point types.
pub trait FloatExt:
    AddAssign + BurnElement + ConstOne + ConstZero + FloatTrait + FromPrimitive
{
    /// The precision of this floating point type
    type Precision: PrecisionSettings;

    /// Convert a usize to a floating point number
    fn from_usize(x: usize) -> Self;
}

impl FloatExt for f32 {
    type Precision = FullPrecisionSettings;

    #[expect(clippy::cast_precision_loss)]
    fn from_usize(x: usize) -> Self {
        x as Self
    }
}

impl FloatExt for f64 {
    type Precision = DoublePrecisionSettings;

    #[expect(clippy::cast_precision_loss)]
    fn from_usize(x: usize) -> Self {
        x as Self
    }
}

#[expect(clippy::similar_names)] // train_xs and train_ys
#[expect(clippy::missing_panics_doc)] // only panics on implementation bugs
#[expect(clippy::too_many_arguments)] // FIXME
/// Encodes the `data` by training a fourier feature neural network.
///
/// The `fourier_features` are randomly sampled from a normal distribution with
/// zero mean and `fourier_scale` standard deviation.
///
/// The neural network consists of `num_blocks` blocks.
///
/// The network is trained for `num_epochs` using the `learning_rate`
/// and mini-batches of `mini_batch_size` if mini-batching is enabled.
///
/// All random numbers are generated using the provided `seed`.
///
/// # Errors
///
/// Errors with
/// - [`FourierNetworkCodecError::NonFiniteData`] if any data element is
///   non-finite (infinite or NaN)
/// - [`FourierNetworkCodecError::NeuralNetworkError`] if an error occurs during
///   the neural network computation
pub fn encode<T: FloatExt, S: Data<Elem = T>, D: Dimension, B: AutodiffBackend<FloatElem = T>>(
    device: &B::Device,
    data: ArrayBase<S, D>,
    fourier_features: NonZeroUsize,
    fourier_scale: Positive<f64>,
    num_blocks: NonZeroUsize,
    learning_rate: Positive<f64>,
    num_epochs: usize,
    mini_batch_size: Option<NonZeroUsize>,
    seed: u64,
) -> Result<Array<u8, Ix1>, FourierNetworkCodecError> {
    let Some(mean) = data.mean() else {
        return Ok(Array::from_vec(Vec::new()));
    };
    let stdv = data.std(T::ZERO);
    let stdv = if stdv == T::ZERO { T::ONE } else { stdv };

    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(FourierNetworkCodecError::NonFiniteData);
    }

    B::seed(seed);

    let b_t = Tensor::<B, 2, Float>::random(
        [data.ndim(), fourier_features.get()],
        Distribution::Normal(0.0, fourier_scale.0),
        device,
    );

    let train_xs = flat_grid_like(&data, device);
    let train_xs = fourier_mapping(train_xs, b_t.clone());

    let train_ys_shape = [data.len(), 1];
    let mut train_ys = data.into_owned();
    train_ys.mapv_inplace(|x| (x - mean) / stdv);
    #[expect(clippy::unwrap_used)] // reshape with one extra new axis cannot fail
    let train_ys = train_ys
        .into_shape_clone((train_ys_shape, Order::RowMajor))
        .unwrap();
    let train_ys = Tensor::from_data(
        TensorData::new(train_ys.into_raw_vec_and_offset().0, train_ys_shape),
        device,
    );

    let model = train(
        device,
        &train_xs,
        &train_ys,
        fourier_features,
        num_blocks,
        learning_rate,
        num_epochs,
        mini_batch_size,
        stdv,
    );

    let extra = ModelExtra {
        model: model.into_record(),
        b_t: Param::from_tensor(b_t).set_require_grad(false),
        mean: Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![mean], vec![1]),
            device,
        ))
        .set_require_grad(false),
        stdv: Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![stdv], vec![1]),
            device,
        ))
        .set_require_grad(false),
        version: StaticCodecVersion,
    };

    let recorder = BinBytesRecorder::<T::Precision>::new();
    let encoded = recorder.record(extra, ()).map_err(NeuralNetworkError)?;

    Ok(Array::from_vec(encoded))
}

#[expect(clippy::missing_panics_doc)] // only panics on implementation bugs
/// Decodes the `encoded` data into the `decoded` output array by making a
/// prediction using the fourier feature neural network.
///
/// The network must have been trained during [`encode`] using the same number
/// of `feature_features` and `num_blocks`.
///
/// # Errors
///
/// Errors with
/// - [`FourierNetworkCodecError::MismatchedDecodeIntoArray`] if the encoded
///   array is empty but the decoded array is not
/// - [`FourierNetworkCodecError::NeuralNetworkError`] if an error occurs during
///   the neural network computation
pub fn decode_into<T: FloatExt, S: Data<Elem = u8>, D: Dimension, B: Backend<FloatElem = T>>(
    device: &B::Device,
    encoded: ArrayBase<S, Ix1>,
    mut decoded: ArrayViewMut<T, D>,
    fourier_features: NonZeroUsize,
    num_blocks: NonZeroUsize,
) -> Result<(), FourierNetworkCodecError> {
    if encoded.is_empty() {
        if decoded.is_empty() {
            return Ok(());
        }

        return Err(FourierNetworkCodecError::MismatchedDecodeIntoArray {
            source: AnyArrayAssignError::ShapeMismatch {
                src: encoded.shape().to_vec(),
                dst: decoded.shape().to_vec(),
            },
        });
    }

    let encoded = encoded.into_owned().into_raw_vec_and_offset().0;

    let recorder = BinBytesRecorder::<T::Precision>::new();
    let record: ModelExtra<B> = recorder.load(encoded, device).map_err(NeuralNetworkError)?;

    let model = ModelConfig::new(fourier_features, num_blocks)
        .init(device)
        .load_record(record.model);
    let b_t = record.b_t.into_value();
    let mean = record.mean.into_value().into_scalar();
    let stdv = record.stdv.into_value().into_scalar();

    let test_xs = flat_grid_like(&decoded, device);
    let test_xs = fourier_mapping(test_xs, b_t);

    let prediction = model.forward(test_xs).into_data();
    #[expect(clippy::unwrap_used)] // same generic type, check must succeed
    let prediction = prediction.as_slice().unwrap();

    #[expect(clippy::unwrap_used)] // prediction shape is flattened
    decoded.assign(&ArrayView::from_shape(decoded.shape(), prediction).unwrap());
    decoded.mapv_inplace(|x| (x * stdv) + mean);

    Ok(())
}

fn flat_grid_like<T: FloatExt, S: Data<Elem = T>, D: Dimension, B: Backend<FloatElem = T>>(
    a: &ArrayBase<S, D>,
    device: &B::Device,
) -> Tensor<B, 2, Float> {
    let grid = a
        .shape()
        .iter()
        .copied()
        .map(|s| {
            #[expect(clippy::useless_conversion)] // (0..s).into_iter()
            (0..s)
                .into_iter()
                .map(move |x| <T as FloatExt>::from_usize(x) / <T as FloatExt>::from_usize(s))
        })
        .multi_cartesian_product()
        .flatten()
        .collect::<Vec<_>>();

    Tensor::from_data(TensorData::new(grid, [a.len(), a.ndim()]), device)
}

fn fourier_mapping<B: Backend>(
    xs: Tensor<B, 2, Float>,
    b_t: Tensor<B, 2, Float>,
) -> Tensor<B, 2, Float> {
    let xs_proj = xs.mul_scalar(core::f64::consts::TAU).matmul(b_t);

    Tensor::cat(vec![xs_proj.clone().sin(), xs_proj.cos()], 1)
}

#[expect(clippy::similar_names)] // train_xs and train_ys
#[expect(clippy::too_many_arguments)] // FIXME
fn train<T: FloatExt, B: AutodiffBackend<FloatElem = T>>(
    device: &B::Device,
    train_xs: &Tensor<B, 2, Float>,
    train_ys: &Tensor<B, 2, Float>,
    fourier_features: NonZeroUsize,
    num_blocks: NonZeroUsize,
    learning_rate: Positive<f64>,
    num_epochs: usize,
    mini_batch_size: Option<NonZeroUsize>,
    stdv: T,
) -> Model<B> {
    let num_samples = train_ys.shape().num_elements();
    let num_batches = mini_batch_size.map(|b| num_samples.div_ceil(b.get()));

    let mut model = ModelConfig::new(fourier_features, num_blocks).init(device);
    let mut optim = AdamConfig::new().init();

    let mut best_loss = T::infinity();
    let mut best_epoch = 0;
    let mut best_model_checkpoint = model.clone().into_record().into_item::<T::Precision>();

    for epoch in 1..=num_epochs {
        #[expect(clippy::option_if_let_else)]
        let (train_xs_batches, train_ys_batches) = match num_batches {
            Some(num_batches) => {
                let shuffle = Tensor::<B, 1, Float>::random(
                    [num_samples],
                    Distribution::Uniform(0.0, 1.0),
                    device,
                );
                let shuffle_indices = shuffle.argsort(0);

                let train_xs_shuffled = train_xs.clone().select(0, shuffle_indices.clone());
                let train_ys_shuffled = train_ys.clone().select(0, shuffle_indices);

                (
                    train_xs_shuffled.chunk(num_batches, 0),
                    train_ys_shuffled.chunk(num_batches, 0),
                )
            }
            None => (vec![train_xs.clone()], vec![train_ys.clone()]),
        };

        let mut loss_sum = T::ZERO;

        let mut se_sum = T::ZERO;
        let mut ae_sum = T::ZERO;
        let mut l_inf = T::ZERO;

        for (train_xs_batch, train_ys_batch) in train_xs_batches.into_iter().zip(train_ys_batches) {
            let prediction = model.forward(train_xs_batch);
            let loss =
                MseLoss::new().forward(prediction.clone(), train_ys_batch.clone(), Reduction::Mean);

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(learning_rate.0, model, grads);

            loss_sum += loss.into_scalar();

            let err = prediction - train_ys_batch;

            se_sum += (err.clone() * err.clone()).sum().into_scalar();
            ae_sum += err.clone().abs().sum().into_scalar();
            l_inf = l_inf.max(err.abs().max().into_scalar());
        }

        let loss_mean = loss_sum / <T as FloatExt>::from_usize(num_batches.unwrap_or(1));

        if loss_mean < best_loss {
            best_loss = loss_mean;
            best_epoch = epoch;
            best_model_checkpoint = model.clone().into_record().into_item::<T::Precision>();
        }

        let rmse = stdv * (se_sum / <T as FloatExt>::from_usize(num_samples)).sqrt();
        let mae = stdv * ae_sum / <T as FloatExt>::from_usize(num_samples);
        let l_inf = stdv * l_inf;

        log::info!(
            "[{epoch}/{num_epochs}]: loss={loss_mean:0.3} MAE={mae:0.3} RMSE={rmse:0.3} Linf={l_inf:0.3}"
        );
    }

    if best_epoch != num_epochs {
        model = model.load_record(ModelRecord::from_item(best_model_checkpoint, device));

        log::info!("restored from epoch {best_epoch} with lowest loss={best_loss:0.3}");
    }

    model
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        std::mem::drop(simple_logger::init());

        let encoded = encode::<f32, _, _, Autodiff<NdArray<f32>>>(
            &NdArrayDevice::Cpu,
            Array::<f32, _>::zeros((0,)),
            NonZeroUsize::MIN,
            Positive(1.0),
            NonZeroUsize::MIN,
            Positive(1e-4),
            10,
            None,
            42,
        )
        .unwrap();
        assert!(encoded.is_empty());
        let mut decoded = Array::<f32, _>::zeros((0,));
        decode_into::<f32, _, _, NdArray<f32>>(
            &NdArrayDevice::Cpu,
            encoded,
            decoded.view_mut(),
            NonZeroUsize::MIN,
            NonZeroUsize::MIN,
        )
        .unwrap();
    }

    #[test]
    fn ones() {
        std::mem::drop(simple_logger::init());

        let encoded = encode::<f32, _, _, Autodiff<NdArray<f32>>>(
            &NdArrayDevice::Cpu,
            Array::<f32, _>::zeros((1, 1, 1, 1)),
            NonZeroUsize::MIN,
            Positive(1.0),
            NonZeroUsize::MIN,
            Positive(1e-4),
            10,
            None,
            42,
        )
        .unwrap();
        let mut decoded = Array::<f32, _>::zeros((1, 1, 1, 1));
        decode_into::<f32, _, _, NdArray<f32>>(
            &NdArrayDevice::Cpu,
            encoded,
            decoded.view_mut(),
            NonZeroUsize::MIN,
            NonZeroUsize::MIN,
        )
        .unwrap();
    }

    #[test]
    fn r#const() {
        std::mem::drop(simple_logger::init());

        let encoded = encode::<f32, _, _, Autodiff<NdArray<f32>>>(
            &NdArrayDevice::Cpu,
            Array::<f32, _>::from_elem((2, 1, 3), 42.0),
            NonZeroUsize::MIN,
            Positive(1.0),
            NonZeroUsize::MIN,
            Positive(1e-4),
            10,
            None,
            42,
        )
        .unwrap();
        let mut decoded = Array::<f32, _>::zeros((2, 1, 3));
        decode_into::<f32, _, _, NdArray<f32>>(
            &NdArrayDevice::Cpu,
            encoded,
            decoded.view_mut(),
            NonZeroUsize::MIN,
            NonZeroUsize::MIN,
        )
        .unwrap();
    }

    #[test]
    fn const_batched() {
        std::mem::drop(simple_logger::init());

        let encoded = encode::<f32, _, _, Autodiff<NdArray<f32>>>(
            &NdArrayDevice::Cpu,
            Array::<f32, _>::from_elem((2, 1, 3), 42.0),
            NonZeroUsize::MIN,
            Positive(1.0),
            NonZeroUsize::MIN,
            Positive(1e-4),
            10,
            Some(NonZeroUsize::MIN.saturating_add(1)),
            42,
        )
        .unwrap();
        let mut decoded = Array::<f32, _>::zeros((2, 1, 3));
        decode_into::<f32, _, _, NdArray<f32>>(
            &NdArrayDevice::Cpu,
            encoded,
            decoded.view_mut(),
            NonZeroUsize::MIN,
            NonZeroUsize::MIN,
        )
        .unwrap();
    }

    #[test]
    fn linspace() {
        std::mem::drop(simple_logger::init());

        let data = Array::linspace(0.0_f64, 100.0_f64, 100);

        let fourier_features = NonZeroUsize::new(16).unwrap();
        let fourier_scale = Positive(10.0);
        let num_blocks = NonZeroUsize::new(2).unwrap();
        let learning_rate = Positive(1e-4);
        let num_epochs = 100;
        let seed = 42;

        for mini_batch_size in [
            None,                                         // no mini-batching
            Some(NonZeroUsize::MIN),                      // stochastic
            Some(NonZeroUsize::MIN.saturating_add(6)),    // mini-batched, remainder
            Some(NonZeroUsize::MIN.saturating_add(9)),    // mini-batched
            Some(NonZeroUsize::MIN.saturating_add(1000)), // mini-batched, truncated
        ] {
            let mut decoded = Array::<f64, _>::zeros(data.shape());
            let encoded = encode::<f64, _, _, Autodiff<NdArray<f64>>>(
                &NdArrayDevice::Cpu,
                data.view(),
                fourier_features,
                fourier_scale,
                num_blocks,
                learning_rate,
                num_epochs,
                mini_batch_size,
                seed,
            )
            .unwrap();

            decode_into::<f64, _, _, NdArray<f64>>(
                &NdArrayDevice::Cpu,
                encoded,
                decoded.view_mut(),
                fourier_features,
                num_blocks,
            )
            .unwrap();
        }
    }
}
