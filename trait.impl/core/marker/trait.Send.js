(function() {var implementors = {
"numcodecs":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayDType.html\" title=\"enum numcodecs::AnyArrayDType\">AnyArrayDType</a>",1,["numcodecs::array::AnyArrayDType"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    &lt;T as <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div>",1,["numcodecs::array::AnyArrayBase"]],["impl&lt;T&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs/struct.StaticCodecType.html\" title=\"struct numcodecs::StaticCodecType\">StaticCodecType</a>&lt;T&gt;",1,["numcodecs::codec::StaticCodecType"]]],
"numcodecs_bit_round":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"numcodecs_bit_round/enum.BitRoundCodecError.html\" title=\"enum numcodecs_bit_round::BitRoundCodecError\">BitRoundCodecError</a>",1,["numcodecs_bit_round::BitRoundCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_bit_round/struct.BitRoundCodec.html\" title=\"struct numcodecs_bit_round::BitRoundCodec\">BitRoundCodec</a>",1,["numcodecs_bit_round::BitRoundCodec"]]],
"numcodecs_identity":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"numcodecs_identity/enum.IdentityCodecError.html\" title=\"enum numcodecs_identity::IdentityCodecError\">IdentityCodecError</a>",1,["numcodecs_identity::IdentityCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_identity/struct.IdentityCodec.html\" title=\"struct numcodecs_identity::IdentityCodec\">IdentityCodec</a>",1,["numcodecs_identity::IdentityCodec"]]],
"numcodecs_python":[["impl !<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodec.html\" title=\"struct numcodecs_python::PyCodec\">PyCodec</a>",1,["numcodecs_python::codec::PyCodec"]],["impl !<a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecClass.html\" title=\"struct numcodecs_python::PyCodecClass\">PyCodecClass</a>",1,["numcodecs_python::codec_class::PyCodecClass"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecAdapter.html\" title=\"struct numcodecs_python::PyCodecAdapter\">PyCodecAdapter</a>",1,["numcodecs_python::adapter::PyCodecAdapter"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecClassAdapter.html\" title=\"struct numcodecs_python::PyCodecClassAdapter\">PyCodecClassAdapter</a>",1,["numcodecs_python::adapter::PyCodecClassAdapter"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_python/struct.PyCodecRegistry.html\" title=\"struct numcodecs_python::PyCodecRegistry\">PyCodecRegistry</a>",1,["numcodecs_python::registry::PyCodecRegistry"]]],
"numcodecs_uniform_noise":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"numcodecs_uniform_noise/enum.UniformNoiseCodecError.html\" title=\"enum numcodecs_uniform_noise::UniformNoiseCodecError\">UniformNoiseCodecError</a>",1,["numcodecs_uniform_noise::UniformNoiseCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_uniform_noise/struct.UniformNoiseCodec.html\" title=\"struct numcodecs_uniform_noise::UniformNoiseCodec\">UniformNoiseCodec</a>",1,["numcodecs_uniform_noise::UniformNoiseCodec"]]],
"numcodecs_zlib":[["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"numcodecs_zlib/enum.ZlibCodecError.html\" title=\"enum numcodecs_zlib::ZlibCodecError\">ZlibCodecError</a>",1,["numcodecs_zlib::ZlibCodecError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"enum\" href=\"numcodecs_zlib/enum.ZlibLevel.html\" title=\"enum numcodecs_zlib::ZlibLevel\">ZlibLevel</a>",1,["numcodecs_zlib::ZlibLevel"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_zlib/struct.ZlibCodec.html\" title=\"struct numcodecs_zlib::ZlibCodec\">ZlibCodec</a>",1,["numcodecs_zlib::ZlibCodec"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_zlib/struct.ZlibDecodeError.html\" title=\"struct numcodecs_zlib::ZlibDecodeError\">ZlibDecodeError</a>",1,["numcodecs_zlib::ZlibDecodeError"]],["impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> for <a class=\"struct\" href=\"numcodecs_zlib/struct.ZlibHeaderError.html\" title=\"struct numcodecs_zlib::ZlibHeaderError\">ZlibHeaderError</a>",1,["numcodecs_zlib::ZlibHeaderError"]]]
};if (window.register_implementors) {window.register_implementors(implementors);} else {window.pending_implementors = implementors;}})()