use std::num::NonZeroUsize;

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{BatchNorm, BatchNormConfig, Gelu, Linear, LinearConfig},
    prelude::Backend,
    record::{PrecisionSettings, Record},
    tensor::{Float, Tensor},
};

#[derive(Debug, Module)]
pub struct Block<B: Backend> {
    bn2_1: BatchNorm<B, 0>,
    gu2_2: Gelu,
    ln2_3: Linear<B>,
}

impl<B: Backend> Block<B> {
    #[expect(clippy::let_and_return)]
    pub fn forward(&self, x: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        let x = self.bn2_1.forward(x);
        let x = self.gu2_2.forward(x);
        let x = self.ln2_3.forward(x);
        x
    }
}

#[derive(Config, Debug)]
pub struct BlockConfig {
    pub fourier_features: NonZeroUsize,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Block<B> {
        Block {
            bn2_1: BatchNormConfig::new(self.fourier_features.get()).init(device),
            gu2_2: Gelu,
            ln2_3: LinearConfig::new(self.fourier_features.get(), self.fourier_features.get())
                .init(device),
        }
    }
}

#[derive(Debug, Module)]
pub struct Model<B: Backend> {
    ln1: Linear<B>,
    bl2: Vec<Block<B>>,
    bn3: BatchNorm<B, 0>,
    gu4: Gelu,
    ln5: Linear<B>,
}

impl<B: Backend> Model<B> {
    #[expect(clippy::let_and_return)]
    pub fn forward(&self, x: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
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
pub struct ModelConfig {
    pub fourier_features: NonZeroUsize,
    pub num_blocks: NonZeroUsize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let block = BlockConfig::new(self.fourier_features);

        Model {
            ln1: LinearConfig::new(self.fourier_features.get() * 2, self.fourier_features.get())
                .init(device),
            #[expect(clippy::useless_conversion)] // (1..num_blocks).into_iter()
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

pub struct ModelExtra<B: Backend> {
    pub model: <Model<B> as Module<B>>::Record,
    pub b_t: Param<Tensor<B, 2, Float>>,
    pub mean: Param<Tensor<B, 1, Float>>,
    pub stdv: Param<Tensor<B, 1, Float>>,
    pub version: crate::FourierNetworkCodecVersion,
}

impl<B: Backend> Record<B> for ModelExtra<B> {
    type Item<S: PrecisionSettings> = ModelExtraItem<B, S>;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        ModelExtraItem {
            model: self.model.into_item(),
            b_t: self.b_t.into_item(),
            mean: self.mean.into_item(),
            stdv: self.stdv.into_item(),
            version: self.version,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        Self {
            model: Record::<B>::from_item::<S>(item.model, device),
            b_t: Record::<B>::from_item::<S>(item.b_t, device),
            mean: Record::<B>::from_item::<S>(item.mean, device),
            stdv: Record::<B>::from_item::<S>(item.stdv, device),
            version: item.version,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct ModelExtraItem<B: Backend, S: PrecisionSettings> {
    model: <<Model<B> as Module<B>>::Record as Record<B>>::Item<S>,
    b_t: <Param<Tensor<B, 2, Float>> as Record<B>>::Item<S>,
    mean: <Param<Tensor<B, 1, Float>> as Record<B>>::Item<S>,
    stdv: <Param<Tensor<B, 1, Float>> as Record<B>>::Item<S>,
    version: crate::FourierNetworkCodecVersion,
}
