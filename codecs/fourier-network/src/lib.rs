//! [![CI Status]][workflow] [![MSRV]][repo] [![Latest Version]][crates.io] [![Rust Doc Crate]][docs.rs] [![Rust Doc Main]][docs]
//!
//! [CI Status]: https://img.shields.io/github/actions/workflow/status/juntyr/numcodecs-rs/ci.yml?branch=main
//! [workflow]: https://github.com/juntyr/numcodecs-rs/actions/workflows/ci.yml?query=branch%3Amain
//!
//! [MSRV]: https://img.shields.io/badge/MSRV-1.76.0-blue
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

use std::{borrow::Cow, num::NonZeroUsize};

use burn::{
    backend::{ndarray::NdArrayDevice, Autodiff, NdArray},
    module::Param,
    nn::loss::Reduction,
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{
        BinBytesRecorder, DoublePrecisionSettings, FullPrecisionSettings, PrecisionSettings,
        Recorder, RecorderError,
    },
    tensor::{backend::AutodiffBackend, Distribution, Element as BurnElement},
};
use itertools::Itertools;
use ndarray::{Array, ArrayBase, ArrayView, ArrayViewMut, Data, Dimension, Ix1, Order, Zip};
use nn::{loss::MseLoss, BatchNorm, BatchNormConfig, Gelu, Linear, LinearConfig};
use num_traits::{ConstOne, ConstZero, Float as FloatTrait, FromPrimitive};
use numcodecs::{
    AnyArray, AnyArrayAssignError, AnyArrayDType, AnyArrayView, AnyArrayViewMut, AnyCowArray,
    Codec, StaticCodec, StaticCodecConfig,
};
use schemars::{json_schema, JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

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
    /// The learning rate for the `AdamW` optimizer
    pub learning_rate: Positive<f64>,
    /// The number of iterations for which the network is trained
    pub training_iterations: usize,
    /// The optional mini-batch size used during training
    ///
    /// Setting the mini-batch size to `None` disables the use of batching,
    /// i.e. the network is trained using one large batch that includes the
    /// full data.
    pub mini_batch_size: Option<NonZeroUsize>,
    /// The seed for the random number generator used during encoding
    pub seed: u64,
}

impl Codec for FourierNetworkCodec {
    type Error = FourierNetworkCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::U8(
                encode::<f32, _, _, Autodiff<NdArray<f32>>>(
                    NdArrayDevice::Cpu,
                    data,
                    self.fourier_features,
                    self.fourier_scale,
                    self.num_blocks,
                    self.learning_rate,
                    self.training_iterations,
                    self.mini_batch_size,
                    self.seed,
                )?
                .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::U8(
                encode::<f64, _, _, Autodiff<NdArray<f64>>>(
                    NdArrayDevice::Cpu,
                    data,
                    self.fourier_features,
                    self.fourier_scale,
                    self.num_blocks,
                    self.learning_rate,
                    self.training_iterations,
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
        if !matches!(encoded.dtype(), AnyArrayDType::F32 | AnyArrayDType::F64) {
            return Err(FourierNetworkCodecError::UnsupportedDtype(encoded.dtype()));
        }

        match (encoded, decoded) {
            (AnyArrayView::U8(encoded), AnyArrayViewMut::F32(decoded)) => {
                #[allow(clippy::option_if_let_else)]
                match encoded.view().into_dimensionality() {
                    Ok(encoded) => decode_into::<f32, _, _, NdArray<f32>>(
                        NdArrayDevice::Cpu,
                        encoded,
                        decoded,
                        self.fourier_features,
                        self.num_blocks,
                    ),
                    Err(_) => Err(FourierNetworkCodecError::EncodedDataNotOneDimensional {
                        shape: encoded.shape().to_vec(),
                    }),
                }
            }
            (AnyArrayView::U8(encoded), AnyArrayViewMut::F64(decoded)) => {
                #[allow(clippy::option_if_let_else)]
                match encoded.view().into_dimensionality() {
                    Ok(encoded) => decode_into::<f64, _, _, NdArray<f64>>(
                        NdArrayDevice::Cpu,
                        encoded,
                        decoded,
                        self.fourier_features,
                        self.num_blocks,
                    ),
                    Err(_) => Err(FourierNetworkCodecError::EncodedDataNotOneDimensional {
                        shape: encoded.shape().to_vec(),
                    }),
                }
            }
            (encoded @ (AnyArrayView::F32(_) | AnyArrayView::F64(_)), decoded) => {
                Err(FourierNetworkCodecError::MismatchedDecodeIntoArray {
                    source: AnyArrayAssignError::DTypeMismatch {
                        src: encoded.dtype(),
                        dst: decoded.dtype(),
                    },
                })
            }
            (encoded, _) => Err(FourierNetworkCodecError::UnsupportedDtype(encoded.dtype())),
        }
    }
}

impl StaticCodec for FourierNetworkCodec {
    const CODEC_ID: &'static str = "fourier-network";

    type Config<'de> = Self;

    fn from_config(config: Self::Config<'_>) -> Self {
        config
    }

    fn get_config(&self) -> StaticCodecConfig<Self> {
        StaticCodecConfig::from(self)
    }
}

#[allow(clippy::derive_partial_eq_without_eq)] // floats are not Eq
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
/// Errors that may occur when applying the [`IdentityCodec`].
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
    /// [`FourierNetworkCodec`] can only decode one-dimensional byte arrays but received
    /// an array of a different shape
    #[error("FourierNetwork can only decode one-dimensional arrays but received an array of shape {shape:?}")]
    EncodedDataNotOneDimensional {
        /// The unexpected shape of the encoded array
        shape: Vec<usize>,
    },
    /// [`FourierNetworkCodec`] cannot decode data of the wrong length
    #[error("FourierNetwork cannot decode data of the wrong length")]
    CorruptedEncodedData,
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
pub trait FloatExt: BurnElement + FloatTrait + FromPrimitive + ConstZero + ConstOne {
    type Precision: PrecisionSettings;

    fn from_usize(x: usize) -> Self;
}

impl FloatExt for f32 {
    type Precision = FullPrecisionSettings;

    fn from_usize(x: usize) -> Self {
        x as Self
    }
}

impl FloatExt for f64 {
    type Precision = DoublePrecisionSettings;

    fn from_usize(x: usize) -> Self {
        x as Self
    }
}

/// Encodes the `data` by training a fourier feature neural network.
///
/// The `fourier_features` are randomly sampled from a normal distribution with
/// zero mean and `fourier_scale` standard deviation.
///
/// The neural network consists of `num_blocks` blocks.
///
/// The network is trained for `training_iterations` using the `learning_rate`
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
    device: B::Device,
    data: ArrayBase<S, D>,
    fourier_features: NonZeroUsize,
    fourier_scale: Positive<f64>,
    num_blocks: NonZeroUsize,
    learning_rate: Positive<f64>,
    training_iterations: usize,
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
        &device,
    );

    let train_xs = flat_grid_like(&data, &device);
    let train_xs = fourier_mapping(train_xs, b_t.clone());

    let train_ys_shape = [data.len(), 1];
    let mut train_ys = data.into_owned();
    train_ys.mapv_inplace(|x| (x - mean) / stdv);
    #[allow(clippy::unwrap_used)] // reshape with one extra new axis cannot fail
    let train_ys = train_ys
        .into_shape_clone((train_ys_shape, Order::RowMajor))
        .unwrap();
    let train_ys = Tensor::from_data(
        TensorData::new(train_ys.into_raw_vec_and_offset().0, train_ys_shape),
        &device,
    );

    let model = train(
        &device,
        train_xs,
        train_ys,
        fourier_features,
        num_blocks,
        learning_rate,
        training_iterations,
        mini_batch_size,
    );

    let extra = ModelExtra {
        model,
        b_t: Param::from_tensor(b_t).set_require_grad(false),
        mean: Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![mean], vec![1]),
            &device,
        ))
        .set_require_grad(false),
        stdv: Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![stdv], vec![1]),
            &device,
        ))
        .set_require_grad(false),
    };

    let recorder = BinBytesRecorder::<T::Precision>::new();
    let encoded = recorder
        .record(extra.into_record(), ())
        .map_err(NeuralNetworkError)?;

    Ok(Array::from_vec(encoded))
}

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
/// - [`FourierNetworkCodecError::CorruptedEncodedData`] if the encoded data is
///   of the wrong length
pub fn decode_into<T: FloatExt, S: Data<Elem = u8>, D: Dimension, B: Backend<FloatElem = T>>(
    device: B::Device,
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
    let record = recorder
        .load(encoded, &device)
        .map_err(NeuralNetworkError)?;

    let extra = ModelExtra::<B> {
        model: ModelConfig::new(fourier_features, num_blocks).init(&device),
        b_t: Param::from_tensor(Tensor::zeros(
            [decoded.ndim(), fourier_features.get()],
            &device,
        ))
        .set_require_grad(false),
        mean: Param::from_tensor(Tensor::zeros([1], &device)).set_require_grad(false),
        stdv: Param::from_tensor(Tensor::ones([1], &device)).set_require_grad(false),
    }
    .load_record(record);

    let model = extra.model;
    let b_t = extra.b_t.into_value();
    let mean = extra.mean.into_value().into_scalar();
    let stdv = extra.stdv.into_value().into_scalar();

    let test_xs = flat_grid_like(&decoded, &device);
    let test_xs = fourier_mapping(test_xs, b_t);

    let prediction = model.forward(test_xs).into_data();
    let prediction = prediction.as_slice().unwrap();

    #[allow(clippy::unwrap_used)] // prediction shape is flattened
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

#[derive(Debug, Module)]
struct Block<B: Backend> {
    bn2_1: BatchNorm<B, 0>,
    gu2_2: Gelu,
    ln2_3: Linear<B>,
}

impl<B: Backend> Block<B> {
    fn forward(&self, x: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let x = self.bn2_1.forward(x);
        let x = self.gu2_2.forward(x);
        let x = self.ln2_3.forward(x);
        x
    }
}

#[derive(Config, Debug)]
struct BlockConfig {
    fourier_features: NonZeroUsize,
}

impl BlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            bn2_1: BatchNormConfig::new(self.fourier_features.get()).init(device),
            gu2_2: Gelu,
            ln2_3: LinearConfig::new(self.fourier_features.get(), self.fourier_features.get())
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
struct Model<B: Backend> {
    ln1: Linear<B>,
    bl2: Vec<Block<B>>,
    bn3: BatchNorm<B, 0>,
    gu4: Gelu,
    ln5: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn forward(&self, x: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let x = self.ln1.forward(x);

        let mut x = x;
        for block in &self.bl2 {
            x = block.forward(x);
        }

        let x = self.bn3.forward(x);
        let x = self.gu4.forward(x);
        let x = self.ln5.forward(x);

        x
    }
}

#[derive(Config, Debug)]
struct ModelConfig {
    fourier_features: NonZeroUsize,
    num_blocks: NonZeroUsize,
}

impl ModelConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let block = BlockConfig::new(self.fourier_features);

        Model {
            ln1: LinearConfig::new(self.fourier_features.get() * 2, self.fourier_features.get())
                .init(device),
            bl2: (1..self.num_blocks.get())
                .into_iter()
                .map(|_| block.init(device))
                .collect(),
            bn3: BatchNormConfig::new(self.fourier_features.get()).init(device),
            gu4: Gelu,
            ln5: LinearConfig::new(self.fourier_features.get(), 1).init(device),
        }
    }
}

#[derive(Debug, Module)]
struct ModelExtra<B: Backend> {
    model: Model<B>,
    b_t: Param<Tensor<B, 2, Float>>,
    mean: Param<Tensor<B, 1, Float>>,
    stdv: Param<Tensor<B, 1, Float>>,
}

#[derive(Config)]
struct TrainingConfig {
    num_epochs: usize,
    batch_size: usize,
    seed: u64,
    learning_rate: f64,
    model: ModelConfig,
    optimizer: AdamConfig,
}

fn train<B: AutodiffBackend>(
    device: &B::Device,
    train_xs: Tensor<B, 2, Float>,
    train_ys: Tensor<B, 2, Float>,
    fourier_features: NonZeroUsize,
    num_blocks: NonZeroUsize,
    learning_rate: Positive<f64>,
    training_iterations: usize,
    mini_batch_size: Option<NonZeroUsize>,
) -> Model<B> {
    // Create the configuration.
    let config_model = ModelConfig::new(fourier_features, num_blocks);
    let config_optimizer = AdamConfig::new();
    let config = TrainingConfig::new(
        training_iterations,
        mini_batch_size.map_or(0, NonZeroUsize::get),
        43,
        learning_rate.0,
        config_model,
        config_optimizer,
    );

    // Create the model and optimizer.
    let mut model = config.model.init(device);
    let mut optim = config.optimizer.init();

    // Iterate over our training and validation loop for X epochs.
    for epoch in 1..config.num_epochs + 1 {
        let output = model.forward(train_xs.clone());
        let loss = MseLoss::new().forward(output.clone(), train_ys.clone(), Reduction::Mean);

        // Gradients for the current backward pass
        let grads = loss.backward();
        // Gradients linked to each parameter of the model.
        let grads = GradientsParams::from_grads(grads, &model);
        // Update the model using the optimizer.
        model = optim.step(config.learning_rate, model, grads);

        log::info!(
            "[{epoch}/{training_iterations}]: loss={}",
            loss.into_scalar(),
        );
    }

    model
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        std::mem::drop(simple_logger::init());

        let encoded = encode::<f32, _, _, Autodiff<NdArray<f32>>>(
            NdArrayDevice::Cpu,
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
            NdArrayDevice::Cpu,
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
            NdArrayDevice::Cpu,
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
            NdArrayDevice::Cpu,
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
            NdArrayDevice::Cpu,
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
            NdArrayDevice::Cpu,
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
            NdArrayDevice::Cpu,
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
            NdArrayDevice::Cpu,
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
        let training_iterations = 100;
        let seed = 42;

        for mini_batch_size in [
            None,                                         // no mini-batching
            Some(NonZeroUsize::MIN),                      // stochastic
            Some(NonZeroUsize::MIN.saturating_add(9)),    // mini-batched
            Some(NonZeroUsize::MIN.saturating_add(1000)), // mini-batched, truncated
        ] {
            let mut decoded = Array::<f64, _>::zeros(data.shape());
            let encoded = encode::<f64, _, _, Autodiff<NdArray<f64>>>(
                NdArrayDevice::Cpu,
                data.view(),
                fourier_features,
                fourier_scale,
                num_blocks,
                learning_rate,
                training_iterations,
                mini_batch_size,
                seed,
            )
            .unwrap();

            decode_into::<f64, _, _, NdArray<f64>>(
                NdArrayDevice::Cpu,
                encoded,
                decoded.view_mut(),
                fourier_features,
                num_blocks,
            )
            .unwrap();
        }
    }
}
