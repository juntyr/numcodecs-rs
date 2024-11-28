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

#![allow(clippy::multiple_crate_versions)] // bitflags

use std::{borrow::Cow, num::NonZeroUsize};

use candle_core::{Device, Error as CandleError, FloatDType, Shape, Tensor, WithDType, D};
use candle_nn::{
    linear_b, seq, AdamW, Module, Optimizer, ParamsAdamW, Sequential, VarBuilder, VarMap,
};
use ndarray::{Array, ArrayBase, ArrayViewMut, Data, Dimension, Ix1, Order, Zip};
use num_traits::{ConstOne, ConstZero, Float, FromPrimitive};
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
    fourier_features: NonZeroUsize,
    /// The standard deviation of the Fourier features
    fourier_scale: Positive<f64>,
    /// The number of blocks in the network
    num_blocks: NonZeroUsize,
    /// The learning rate for the `AdamW` optimizer
    learning_rate: Positive<f64>,
    /// The number of iterations for which the network is trained
    training_iterations: usize,
}

impl Codec for FourierNetworkCodec {
    type Error = FourierNetworkCodecError;

    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, Self::Error> {
        match data {
            AnyCowArray::F32(data) => Ok(AnyArray::F32(
                encode(
                    data,
                    self.fourier_features,
                    self.fourier_scale,
                    self.num_blocks,
                    self.learning_rate,
                    self.training_iterations,
                )?
                .into_dyn(),
            )),
            AnyCowArray::F64(data) => Ok(AnyArray::F64(
                encode(
                    data,
                    self.fourier_features,
                    self.fourier_scale,
                    self.num_blocks,
                    self.learning_rate,
                    self.training_iterations,
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
            (AnyArrayView::F32(encoded), AnyArrayViewMut::F32(decoded)) => {
                #[allow(clippy::option_if_let_else)]
                match encoded.view().into_dimensionality() {
                    Ok(encoded) => {
                        decode_into(encoded, decoded, self.fourier_features, self.num_blocks)
                    }
                    Err(_) => Err(FourierNetworkCodecError::EncodedDataNotOneDimensional {
                        shape: encoded.shape().to_vec(),
                    }),
                }
            }
            (AnyArrayView::F64(encoded), AnyArrayViewMut::F64(decoded)) => {
                #[allow(clippy::option_if_let_else)]
                match encoded.view().into_dimensionality() {
                    Ok(encoded) => {
                        decode_into(encoded, decoded, self.fourier_features, self.num_blocks)
                    }
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
pub struct Positive<T: Float>(T);

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
pub struct NeuralNetworkError(CandleError);

/// Floating point types.
pub trait FloatExt: FloatDType + Float + FromPrimitive + ConstZero + ConstOne {}

impl FloatExt for f32 {}
impl FloatExt for f64 {}

#[allow(clippy::similar_names)] // train_xs and train_ys
#[allow(clippy::significant_drop_tightening)] // lock while accessing data
#[allow(clippy::missing_panics_doc)] // only when poisoned
/// Encodes the `data` by training a fourier feature neural network.
///
/// The `fourier_features` are randomly sampled from a normal distribution with
/// zero mean and `fourier_scale` standard deviation.
///
/// The neural network consists of `num_blocks` blocks.
///
/// The network is trained for `training_iterations` using the `learning_rate`.
///
/// # Errors
///
/// Errors with
/// - [`FourierNetworkCodecError::NonFiniteData`] if any data element is
///   non-finite (infinite or NaN)
/// - [`FourierNetworkCodecError::NeuralNetworkError`] if an error occurs during
///   the neural network computation
pub fn encode<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
    data: ArrayBase<S, D>,
    fourier_features: NonZeroUsize,
    fourier_scale: Positive<f64>,
    num_blocks: NonZeroUsize,
    learning_rate: Positive<f64>,
    training_iterations: usize,
) -> Result<Array<T, Ix1>, FourierNetworkCodecError> {
    let Some(mean) = data.mean() else {
        return Ok(Array::from_vec(Vec::new()));
    };
    let stdv = data.std(T::ZERO);
    let stdv = if stdv == T::ZERO { T::ONE } else { stdv };

    if !Zip::from(&data).all(|x| x.is_finite()) {
        return Err(FourierNetworkCodecError::NonFiniteData);
    }

    let b_t = Tensor::randn(
        T::ZERO,
        <T as WithDType>::from_f64(fourier_scale.0),
        (data.ndim(), fourier_features.get()),
        &Device::Cpu,
    )
    .map_err(NeuralNetworkError)?;

    let train_xs = fourier_mapping(
        &stacked_meshgrid::<T>(data.shape())
            .map_err(NeuralNetworkError)?
            .reshape(((), data.ndim()))
            .map_err(NeuralNetworkError)?,
        &b_t,
    )
    .map_err(NeuralNetworkError)?;

    let train_ys_shape = Shape::from_dims(data.shape()).extend(&[1]);
    let mut train_ys = data.into_owned();
    train_ys.mapv_inplace(|x| (x - mean) / stdv);
    #[allow(clippy::unwrap_used)] // reshape with one extra new axis cannot fail
    let train_ys = train_ys
        .into_shape_clone((train_ys_shape.dims(), Order::RowMajor))
        .unwrap();
    let train_ys = Tensor::from_vec(
        train_ys.into_raw_vec_and_offset().0,
        train_ys_shape,
        &Device::Cpu,
    )
    .map_err(NeuralNetworkError)?
    .reshape(((), 1))
    .map_err(NeuralNetworkError)?;

    let model = train::<T>(
        &train_xs,
        &train_ys,
        fourier_features,
        num_blocks,
        learning_rate,
        training_iterations,
    )
    .map_err(NeuralNetworkError)?;

    #[allow(clippy::unwrap_used)] // poisoning is unrecoverable
    let guard = model.data().lock().unwrap();

    let mut vars = guard
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect::<Vec<_>>();
    vars.sort_by(|x, y| x.0.cmp(&y.0));

    let mut encoded = Vec::new();
    for (_, var) in vars {
        encoded.append(
            &mut var
                .flatten_all()
                .map_err(NeuralNetworkError)?
                .to_vec1()
                .map_err(NeuralNetworkError)?,
        );
    }
    encoded.append(
        &mut b_t
            .flatten_all()
            .map_err(NeuralNetworkError)?
            .to_vec1()
            .map_err(NeuralNetworkError)?,
    );
    encoded.push(mean);
    encoded.push(stdv);

    Ok(Array::from_vec(encoded))
}

#[allow(clippy::significant_drop_tightening)] // lock while accessing data
#[allow(clippy::missing_panics_doc)] // only when poisoned
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
/// - [`FourierNetworkCodecError::NonFiniteData`] if the `projected` array or
///   the reconstructed output contains non-finite data
pub fn decode_into<T: FloatExt, S: Data<Elem = T>, D: Dimension>(
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

    let varmap = VarMap::new();
    let model = make_model::<T>(num_blocks, fourier_features, &varmap, false)
        .map_err(NeuralNetworkError)?;

    let mut i = 0;

    {
        #[allow(clippy::unwrap_used)] // poisoning is unrecoverable
        let guard = varmap.data().lock().unwrap();

        let mut vars = guard
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>();
        vars.sort_by(|x, y| x.0.cmp(&y.0));

        for (_, var) in vars {
            var.set(
                &Tensor::from_slice(
                    encoded
                        .get(i..i + var.shape().elem_count())
                        .ok_or(FourierNetworkCodecError::CorruptedEncodedData)?,
                    var.shape().dims(),
                    &Device::Cpu,
                )
                .map_err(NeuralNetworkError)?,
            )
            .map_err(NeuralNetworkError)?;
            i += var.shape().elem_count();
        }
    }

    let b_t = Tensor::from_slice(
        encoded
            .get(i..i + decoded.ndim() * fourier_features.get())
            .ok_or(FourierNetworkCodecError::CorruptedEncodedData)?,
        &[decoded.ndim(), fourier_features.get()],
        &Device::Cpu,
    )
    .map_err(NeuralNetworkError)?;
    i += decoded.ndim() * fourier_features.get();

    let mean = *encoded
        .get(i)
        .ok_or(FourierNetworkCodecError::CorruptedEncodedData)?;
    let stdv = *encoded
        .get(i + 1)
        .ok_or(FourierNetworkCodecError::CorruptedEncodedData)?;

    if encoded.len() != (i + 2) {
        return Err(FourierNetworkCodecError::CorruptedEncodedData);
    }

    let test_xs = fourier_mapping(
        &stacked_meshgrid::<T>(decoded.shape())
            .map_err(NeuralNetworkError)?
            .reshape(((), decoded.ndim()))
            .map_err(NeuralNetworkError)?,
        &b_t,
    )
    .map_err(NeuralNetworkError)?;

    let prediction = model.forward(&test_xs).map_err(NeuralNetworkError)?;
    let prediction = prediction
        .flatten_all()
        .map_err(NeuralNetworkError)?
        .to_vec1()
        .map_err(NeuralNetworkError)?;

    #[allow(clippy::unwrap_used)] // prediction shape is flattened
    decoded.assign(&Array::from_shape_vec(decoded.shape(), prediction).unwrap());
    decoded.mapv_inplace(|x| (x * stdv) + mean);

    if !Zip::from(decoded).all(|x| x.is_finite()) {
        return Err(FourierNetworkCodecError::NonFiniteData);
    }

    Ok(())
}

fn stacked_meshgrid<T: FloatExt>(shape: &[usize]) -> Result<Tensor, CandleError> {
    let axes = shape
        .iter()
        .map(|s| {
            #[allow(clippy::cast_precision_loss)]
            Tensor::arange(T::ZERO, <T as WithDType>::from_f64(*s as f64), &Device::Cpu)
                .and_then(|t| t / (*s as f64))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if let [a] = axes.as_slice() {
        return a.reshape(a.shape().clone().extend(&[1]));
    }

    Tensor::stack(
        Tensor::meshgrid(axes.as_slice(), false)?.as_slice(),
        D::Minus1,
    )
}

fn fourier_mapping(xs: &Tensor, b_t: &Tensor) -> Result<Tensor, CandleError> {
    let xs_proj =
        (xs.contiguous()? * core::f64::consts::TAU)?.broadcast_matmul(&b_t.contiguous()?)?;

    Tensor::cat(&[xs_proj.sin()?, xs_proj.cos()?], D::Minus1)
}

fn make_model<T: FloatExt>(
    num_blocks: NonZeroUsize,
    fourier_features: NonZeroUsize,
    varmap: &VarMap,
    _train: bool, // FIXME
) -> Result<Sequential, CandleError> {
    let vb = VarBuilder::from_varmap(varmap, T::DTYPE, &Device::Cpu);

    let mut layers = seq();

    let ln1 = linear_b(
        fourier_features.get() * 2,
        fourier_features.get(),
        true,
        vb.pp("ln0"),
    )?;
    layers = layers.add(ln1);

    for _ in 1..num_blocks.get() {
        // let bn2_1 = batch_norm(fourier_features.get(), BatchNormConfig::default(), vb.pp("bn2.1"))?;
        let ln2_2 = linear_b(
            fourier_features.get(),
            fourier_features.get(),
            true,
            vb.pp("ln2.2"),
        )?;
        // let bn2_3 = batch_norm(fourier_features.get(), BatchNormConfig::default(), vb.pp("bn2.3"))?;
        let ln2_4 = linear_b(
            fourier_features.get(),
            fourier_features.get(),
            true,
            vb.pp("ln2.4"),
        )?;

        layers = layers.add_fn(move |xs| {
            xs /*.apply_t(&bn2_1, train)*/
                .gelu()?
                .apply(&ln2_2)? /*.apply_t(&bn2_3, train)?*/
                .gelu()?
                .apply(&ln2_4)
        });
    }

    // let bn3 = batch_norm(fourier_features.get(), BatchNormConfig::default(), vb.pp("bn3"))?;
    layers = layers.add_fn(move |xs| {
        xs /*.apply_t(&bn3, false)?*/
            .gelu()
    });

    let ln4 = linear_b(fourier_features.get(), 1, true, vb.pp("ln4"))?;
    layers = layers.add(ln4);

    Ok(layers)
}

#[allow(clippy::similar_names)]
fn train<T: FloatExt>(
    train_xs: &Tensor,
    train_ys: &Tensor,
    fourier_features: NonZeroUsize,
    num_blocks: NonZeroUsize,
    learning_rate: Positive<f64>,
    training_iterations: usize,
) -> Result<VarMap, CandleError> {
    let varmap = VarMap::new();

    let model = make_model::<T>(num_blocks, fourier_features, &varmap, true)?;

    let mut opt = AdamW::new(
        varmap.all_vars(),
        ParamsAdamW {
            lr: learning_rate.0,
            ..Default::default()
        },
    )?;

    for i in 0..training_iterations {
        let predict_ys = model.forward(train_xs)?;

        let loss = predict_ys.sub(train_ys)?.sqr()?.mean_all()? * 0.5;
        let loss = loss?;

        opt.backward_step(&loss)?;

        log::info!(
            "[{i}/{training_iterations}]: loss={}",
            loss.to_scalar::<T>()?
        );
    }

    Ok(varmap)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn empty() {
        std::mem::drop(simple_logger::init());

        let encoded = encode(
            Array::<f32, _>::zeros((0,)),
            NonZeroUsize::MIN,
            Positive(1.0),
            NonZeroUsize::MIN,
            Positive(1e-4),
            10,
        )
        .unwrap();
        assert!(encoded.is_empty());
        let mut decoded = Array::<f32, _>::zeros((0,));
        decode_into(
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

        let encoded = encode(
            Array::<f32, _>::zeros((1, 1, 1, 1)),
            NonZeroUsize::MIN,
            Positive(1.0),
            NonZeroUsize::MIN,
            Positive(1e-4),
            10,
        )
        .unwrap();
        let mut decoded = Array::<f32, _>::zeros((1, 1, 1, 1));
        decode_into(
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

        let encoded = encode(
            Array::<f32, _>::from_elem((2, 1, 3), 42.0),
            NonZeroUsize::MIN,
            Positive(1.0),
            NonZeroUsize::MIN,
            Positive(1e-4),
            10,
        )
        .unwrap();
        let mut decoded = Array::<f32, _>::zeros((2, 1, 3));
        decode_into(
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

        let mut decoded = Array::zeros(data.shape());
        let encoded = encode(
            data,
            fourier_features,
            fourier_scale,
            num_blocks,
            learning_rate,
            training_iterations,
        )
        .unwrap();

        decode_into(encoded, decoded.view_mut(), fourier_features, num_blocks).unwrap();
    }
}
