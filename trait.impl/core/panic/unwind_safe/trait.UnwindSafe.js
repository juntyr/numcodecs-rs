(function() {
    var implementors = Object.fromEntries([["numcodecs",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayAssignError.html\" title=\"enum numcodecs::AnyArrayAssignError\">AnyArrayAssignError</a>",1,["numcodecs::array::AnyArrayAssignError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayDType.html\" title=\"enum numcodecs::AnyArrayDType\">AnyArrayDType</a>",1,["numcodecs::array::AnyArrayDType"]],["impl&lt;'a, T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs/struct.StaticCodecConfig.html\" title=\"struct numcodecs::StaticCodecConfig\">StaticCodecConfig</a>&lt;'a, T&gt;<div class=\"where\">where\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.StaticCodec.html\" title=\"trait numcodecs::StaticCodec\">StaticCodec</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.StaticCodec.html#associatedtype.Config\" title=\"type numcodecs::StaticCodec::Config\">Config</a>&lt;'a&gt;: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,</div>",1,["numcodecs::codec::StaticCodecConfig"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,</div>",1,["numcodecs::array::AnyArrayBase"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs/struct.StaticCodecType.html\" title=\"struct numcodecs::StaticCodecType\">StaticCodecType</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,</div>",1,["numcodecs::codec::StaticCodecType"]]]],["numcodecs_asinh",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_asinh/enum.AsinhCodecError.html\" title=\"enum numcodecs_asinh::AsinhCodecError\">AsinhCodecError</a>",1,["numcodecs_asinh::AsinhCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_asinh/struct.AsinhCodec.html\" title=\"struct numcodecs_asinh::AsinhCodec\">AsinhCodec</a>",1,["numcodecs_asinh::AsinhCodec"]]]],["numcodecs_bit_round",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_bit_round/enum.BitRoundCodecError.html\" title=\"enum numcodecs_bit_round::BitRoundCodecError\">BitRoundCodecError</a>",1,["numcodecs_bit_round::BitRoundCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_bit_round/struct.BitRoundCodec.html\" title=\"struct numcodecs_bit_round::BitRoundCodec\">BitRoundCodec</a>",1,["numcodecs_bit_round::BitRoundCodec"]]]],["numcodecs_fixed_offset_scale",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_fixed_offset_scale/enum.FixedOffsetScaleCodecError.html\" title=\"enum numcodecs_fixed_offset_scale::FixedOffsetScaleCodecError\">FixedOffsetScaleCodecError</a>",1,["numcodecs_fixed_offset_scale::FixedOffsetScaleCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_fixed_offset_scale/struct.FixedOffsetScaleCodec.html\" title=\"struct numcodecs_fixed_offset_scale::FixedOffsetScaleCodec\">FixedOffsetScaleCodec</a>",1,["numcodecs_fixed_offset_scale::FixedOffsetScaleCodec"]]]],["numcodecs_fourier_network",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_fourier_network/enum.FourierNetworkCodecError.html\" title=\"enum numcodecs_fourier_network::FourierNetworkCodecError\">FourierNetworkCodecError</a>",1,["numcodecs_fourier_network::FourierNetworkCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_fourier_network/struct.FourierNetworkCodec.html\" title=\"struct numcodecs_fourier_network::FourierNetworkCodec\">FourierNetworkCodec</a>",1,["numcodecs_fourier_network::FourierNetworkCodec"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_fourier_network/struct.NeuralNetworkError.html\" title=\"struct numcodecs_fourier_network::NeuralNetworkError\">NeuralNetworkError</a>",1,["numcodecs_fourier_network::NeuralNetworkError"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_fourier_network/struct.Positive.html\" title=\"struct numcodecs_fourier_network::Positive\">Positive</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,</div>",1,["numcodecs_fourier_network::Positive"]]]],["numcodecs_identity",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_identity/enum.IdentityCodecError.html\" title=\"enum numcodecs_identity::IdentityCodecError\">IdentityCodecError</a>",1,["numcodecs_identity::IdentityCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_identity/struct.IdentityCodec.html\" title=\"struct numcodecs_identity::IdentityCodec\">IdentityCodec</a>",1,["numcodecs_identity::IdentityCodec"]]]],["numcodecs_linear_quantize",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_linear_quantize/enum.LinearQuantizeBins.html\" title=\"enum numcodecs_linear_quantize::LinearQuantizeBins\">LinearQuantizeBins</a>",1,["numcodecs_linear_quantize::LinearQuantizeBins"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_linear_quantize/enum.LinearQuantizeCodecError.html\" title=\"enum numcodecs_linear_quantize::LinearQuantizeCodecError\">LinearQuantizeCodecError</a>",1,["numcodecs_linear_quantize::LinearQuantizeCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_linear_quantize/enum.LinearQuantizeDType.html\" title=\"enum numcodecs_linear_quantize::LinearQuantizeDType\">LinearQuantizeDType</a>",1,["numcodecs_linear_quantize::LinearQuantizeDType"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_linear_quantize/struct.LinearQuantizeCodec.html\" title=\"struct numcodecs_linear_quantize::LinearQuantizeCodec\">LinearQuantizeCodec</a>",1,["numcodecs_linear_quantize::LinearQuantizeCodec"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_linear_quantize/struct.LinearQuantizeHeaderError.html\" title=\"struct numcodecs_linear_quantize::LinearQuantizeHeaderError\">LinearQuantizeHeaderError</a>",1,["numcodecs_linear_quantize::LinearQuantizeHeaderError"]]]],["numcodecs_log",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_log/enum.LogCodecError.html\" title=\"enum numcodecs_log::LogCodecError\">LogCodecError</a>",1,["numcodecs_log::LogCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_log/struct.LogCodec.html\" title=\"struct numcodecs_log::LogCodec\">LogCodec</a>",1,["numcodecs_log::LogCodec"]]]],["numcodecs_python",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodec.html\" title=\"struct numcodecs_python::PyCodec\">PyCodec</a>",1,["numcodecs_python::codec::PyCodec"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecAdapter.html\" title=\"struct numcodecs_python::PyCodecAdapter\">PyCodecAdapter</a>",1,["numcodecs_python::adapter::PyCodecAdapter"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecClass.html\" title=\"struct numcodecs_python::PyCodecClass\">PyCodecClass</a>",1,["numcodecs_python::codec_class::PyCodecClass"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecClassAdapter.html\" title=\"struct numcodecs_python::PyCodecClassAdapter\">PyCodecClassAdapter</a>",1,["numcodecs_python::adapter::PyCodecClassAdapter"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecRegistry.html\" title=\"struct numcodecs_python::PyCodecRegistry\">PyCodecRegistry</a>",1,["numcodecs_python::registry::PyCodecRegistry"]]]],["numcodecs_random_projection",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_random_projection/enum.RandomProjectionCodecError.html\" title=\"enum numcodecs_random_projection::RandomProjectionCodecError\">RandomProjectionCodecError</a>",1,["numcodecs_random_projection::RandomProjectionCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_random_projection/enum.RandomProjectionKind.html\" title=\"enum numcodecs_random_projection::RandomProjectionKind\">RandomProjectionKind</a>",1,["numcodecs_random_projection::RandomProjectionKind"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_random_projection/enum.RandomProjectionReduction.html\" title=\"enum numcodecs_random_projection::RandomProjectionReduction\">RandomProjectionReduction</a>",1,["numcodecs_random_projection::RandomProjectionReduction"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_random_projection/struct.RandomProjectionCodec.html\" title=\"struct numcodecs_random_projection::RandomProjectionCodec\">RandomProjectionCodec</a>",1,["numcodecs_random_projection::RandomProjectionCodec"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_random_projection/struct.ClosedOpenUnit.html\" title=\"struct numcodecs_random_projection::ClosedOpenUnit\">ClosedOpenUnit</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,</div>",1,["numcodecs_random_projection::ClosedOpenUnit"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_random_projection/struct.OpenClosedUnit.html\" title=\"struct numcodecs_random_projection::OpenClosedUnit\">OpenClosedUnit</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,</div>",1,["numcodecs_random_projection::OpenClosedUnit"]]]],["numcodecs_reinterpret",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_reinterpret/enum.ReinterpretCodecError.html\" title=\"enum numcodecs_reinterpret::ReinterpretCodecError\">ReinterpretCodecError</a>",1,["numcodecs_reinterpret::ReinterpretCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_reinterpret/struct.ReinterpretCodec.html\" title=\"struct numcodecs_reinterpret::ReinterpretCodec\">ReinterpretCodec</a>",1,["numcodecs_reinterpret::ReinterpretCodec"]]]],["numcodecs_round",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_round/enum.RoundCodecError.html\" title=\"enum numcodecs_round::RoundCodecError\">RoundCodecError</a>",1,["numcodecs_round::RoundCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_round/struct.RoundCodec.html\" title=\"struct numcodecs_round::RoundCodec\">RoundCodec</a>",1,["numcodecs_round::RoundCodec"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_round/struct.Positive.html\" title=\"struct numcodecs_round::Positive\">Positive</a>&lt;T&gt;<div class=\"where\">where\n    T: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a>,</div>",1,["numcodecs_round::Positive"]]]],["numcodecs_swizzle_reshape",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_swizzle_reshape/enum.Axis.html\" title=\"enum numcodecs_swizzle_reshape::Axis\">Axis</a>",1,["numcodecs_swizzle_reshape::Axis"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_swizzle_reshape/enum.AxisGroup.html\" title=\"enum numcodecs_swizzle_reshape::AxisGroup\">AxisGroup</a>",1,["numcodecs_swizzle_reshape::AxisGroup"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_swizzle_reshape/enum.SwizzleReshapeCodecError.html\" title=\"enum numcodecs_swizzle_reshape::SwizzleReshapeCodecError\">SwizzleReshapeCodecError</a>",1,["numcodecs_swizzle_reshape::SwizzleReshapeCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_swizzle_reshape/struct.Rest.html\" title=\"struct numcodecs_swizzle_reshape::Rest\">Rest</a>",1,["numcodecs_swizzle_reshape::Rest"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_swizzle_reshape/struct.SwizzleReshapeCodec.html\" title=\"struct numcodecs_swizzle_reshape::SwizzleReshapeCodec\">SwizzleReshapeCodec</a>",1,["numcodecs_swizzle_reshape::SwizzleReshapeCodec"]]]],["numcodecs_sz3",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_sz3/enum.Sz3CodecError.html\" title=\"enum numcodecs_sz3::Sz3CodecError\">Sz3CodecError</a>",1,["numcodecs_sz3::Sz3CodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_sz3/enum.Sz3DType.html\" title=\"enum numcodecs_sz3::Sz3DType\">Sz3DType</a>",1,["numcodecs_sz3::Sz3DType"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_sz3/enum.Sz3Encoder.html\" title=\"enum numcodecs_sz3::Sz3Encoder\">Sz3Encoder</a>",1,["numcodecs_sz3::Sz3Encoder"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_sz3/enum.Sz3ErrorBound.html\" title=\"enum numcodecs_sz3::Sz3ErrorBound\">Sz3ErrorBound</a>",1,["numcodecs_sz3::Sz3ErrorBound"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_sz3/enum.Sz3LosslessCompressor.html\" title=\"enum numcodecs_sz3::Sz3LosslessCompressor\">Sz3LosslessCompressor</a>",1,["numcodecs_sz3::Sz3LosslessCompressor"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_sz3/enum.Sz3Predictor.html\" title=\"enum numcodecs_sz3::Sz3Predictor\">Sz3Predictor</a>",1,["numcodecs_sz3::Sz3Predictor"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_sz3/struct.Sz3Codec.html\" title=\"struct numcodecs_sz3::Sz3Codec\">Sz3Codec</a>",1,["numcodecs_sz3::Sz3Codec"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_sz3/struct.Sz3CodingError.html\" title=\"struct numcodecs_sz3::Sz3CodingError\">Sz3CodingError</a>",1,["numcodecs_sz3::Sz3CodingError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_sz3/struct.Sz3HeaderError.html\" title=\"struct numcodecs_sz3::Sz3HeaderError\">Sz3HeaderError</a>",1,["numcodecs_sz3::Sz3HeaderError"]]]],["numcodecs_uniform_noise",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_uniform_noise/enum.UniformNoiseCodecError.html\" title=\"enum numcodecs_uniform_noise::UniformNoiseCodecError\">UniformNoiseCodecError</a>",1,["numcodecs_uniform_noise::UniformNoiseCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_uniform_noise/struct.UniformNoiseCodec.html\" title=\"struct numcodecs_uniform_noise::UniformNoiseCodec\">UniformNoiseCodec</a>",1,["numcodecs_uniform_noise::UniformNoiseCodec"]]]],["numcodecs_zfp",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_zfp/enum.ZfpCodecError.html\" title=\"enum numcodecs_zfp::ZfpCodecError\">ZfpCodecError</a>",1,["numcodecs_zfp::ZfpCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_zfp/enum.ZfpCompressionMode.html\" title=\"enum numcodecs_zfp::ZfpCompressionMode\">ZfpCompressionMode</a>",1,["numcodecs_zfp::ZfpCompressionMode"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zfp/struct.ZfpCodec.html\" title=\"struct numcodecs_zfp::ZfpCodec\">ZfpCodec</a>",1,["numcodecs_zfp::ZfpCodec"]]]],["numcodecs_zlib",[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_zlib/enum.ZlibCodecError.html\" title=\"enum numcodecs_zlib::ZlibCodecError\">ZlibCodecError</a>",1,["numcodecs_zlib::ZlibCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_zlib/enum.ZlibLevel.html\" title=\"enum numcodecs_zlib::ZlibLevel\">ZlibLevel</a>",1,["numcodecs_zlib::ZlibLevel"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zlib/struct.ZlibCodec.html\" title=\"struct numcodecs_zlib::ZlibCodec\">ZlibCodec</a>",1,["numcodecs_zlib::ZlibCodec"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zlib/struct.ZlibDecodeError.html\" title=\"struct numcodecs_zlib::ZlibDecodeError\">ZlibDecodeError</a>",1,["numcodecs_zlib::ZlibDecodeError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zlib/struct.ZlibHeaderError.html\" title=\"struct numcodecs_zlib::ZlibHeaderError\">ZlibHeaderError</a>",1,["numcodecs_zlib::ZlibHeaderError"]]]],["numcodecs_zstd",[["impl !<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"enum\" href=\"numcodecs_zstd/enum.ZstdCodecError.html\" title=\"enum numcodecs_zstd::ZstdCodecError\">ZstdCodecError</a>",1,["numcodecs_zstd::ZstdCodecError"]],["impl !<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zstd/struct.ZstdCodingError.html\" title=\"struct numcodecs_zstd::ZstdCodingError\">ZstdCodingError</a>",1,["numcodecs_zstd::ZstdCodingError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zstd/struct.ZstdCodec.html\" title=\"struct numcodecs_zstd::ZstdCodec\">ZstdCodec</a>",1,["numcodecs_zstd::ZstdCodec"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zstd/struct.ZstdHeaderError.html\" title=\"struct numcodecs_zstd::ZstdHeaderError\">ZstdHeaderError</a>",1,["numcodecs_zstd::ZstdHeaderError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/panic/unwind_safe/trait.UnwindSafe.html\" title=\"trait core::panic::unwind_safe::UnwindSafe\">UnwindSafe</a> for <a class=\"struct\" href=\"numcodecs_zstd/struct.ZstdLevel.html\" title=\"struct numcodecs_zstd::ZstdLevel\">ZstdLevel</a>",1,["numcodecs_zstd::ZstdLevel"]]]]]);
    if (window.register_implementors) {
        window.register_implementors(implementors);
    } else {
        window.pending_implementors = implementors;
    }
})()
//{"start":57,"fragment_lengths":[7257,736,788,915,1896,781,2134,706,1904,3040,826,1312,1954,3172,848,1076,1792,1800]}