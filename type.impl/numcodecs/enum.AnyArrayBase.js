(function() {var type_impls = {
"numcodecs":[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#117-163\">source</a><a href=\"#impl-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: AnyRawData&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::U8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::F32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::F64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.view\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#132-145\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.view\" class=\"fn\">view</a>(&amp;self) -&gt; <a class=\"type\" href=\"numcodecs/type.AnyArrayView.html\" title=\"type numcodecs::AnyArrayView\">AnyArrayView</a>&lt;'_&gt;</h4></section></summary><div class=\"docblock\"><p>Returns a read-only view of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.cow\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#149-162\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.cow\" class=\"fn\">cow</a>(&amp;self) -&gt; <a class=\"type\" href=\"numcodecs/type.AnyCowArray.html\" title=\"type numcodecs::AnyCowArray\">AnyCowArray</a>&lt;'_&gt;</h4></section></summary><div class=\"docblock\"><p>Returns a copy-on-write view of the array.</p>\n</div></details></div></details>",0,"numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#165-194\">source</a><a href=\"#impl-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: AnyRawData&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::U8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::U16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::U32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::U64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::I8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::I16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::I32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::I64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::F32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::F64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.view_mut\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#180-193\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.view_mut\" class=\"fn\">view_mut</a>(&amp;mut self) -&gt; <a class=\"type\" href=\"numcodecs/type.AnyArrayViewMut.html\" title=\"type numcodecs::AnyArrayViewMut\">AnyArrayViewMut</a>&lt;'_&gt;</h4></section></summary><div class=\"docblock\"><p>Returns a read-write view of the array.</p>\n</div></details></div></details>",0,"numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#35-115\">source</a><a href=\"#impl-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: AnyRawData&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.len\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#37-50\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.len\" class=\"fn\">len</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>Returns the total number of elements in the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.is_empty\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#53-66\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.is_empty\" class=\"fn\">is_empty</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class=\"docblock\"><p>Returns whether the array has any elements.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.dtype\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#69-82\">source</a><h4 class=\"code-header\">pub const fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.dtype\" class=\"fn\">dtype</a>(&amp;self) -&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayDType.html\" title=\"enum numcodecs::AnyArrayDType\">AnyArrayDType</a></h4></section></summary><div class=\"docblock\"><p>Returns the dtype of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.shape\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#85-98\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.shape\" class=\"fn\">shape</a>(&amp;self) -&gt; &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>]</h4></section></summary><div class=\"docblock\"><p>Returns the shape of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.strides\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#101-114\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.strides\" class=\"fn\">strides</a>(&amp;self) -&gt; &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.isize.html\">isize</a>]</h4></section></summary><div class=\"docblock\"><p>Return the strides of the array.</p>\n</div></details></div></details>",0,"numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Clone-for-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#196-223\">source</a><a href=\"#impl-Clone-for-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: AnyRawData&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::U8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::U16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::U32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::U64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::I8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::I16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::I32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::I64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::F32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::F64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#209-222\">source</a><a href=\"#method.clone\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#tymethod.clone\" class=\"fn\">clone</a>(&amp;self) -&gt; Self</h4></section></summary><div class='docblock'>Returns a copy of the value. <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#tymethod.clone\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone_from\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/clone.rs.html#175\">source</a></span><a href=\"#method.clone_from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#method.clone_from\" class=\"fn\">clone_from</a>(&amp;mut self, source: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Self</a>)</h4></section></summary><div class='docblock'>Performs copy-assignment from <code>source</code>. <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#method.clone_from\">Read more</a></div></details></div></details>","Clone","numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#225-252\">source</a><a href=\"#impl-Debug-for-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: AnyRawData&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::U8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::F32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::F64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#238-251\">source</a><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, fmt: &amp;mut <a class=\"struct\" href=\"https://doc.rust-lang.org/nightly/core/fmt/struct.Formatter.html\" title=\"struct core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"type\" href=\"https://doc.rust-lang.org/nightly/core/fmt/type.Result.html\" title=\"type core::fmt::Result\">Result</a></h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialEq-for-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#254-282\">source</a><a href=\"#impl-PartialEq-for-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: AnyRawData&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::U8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::U64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I8: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I16: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::I64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::F32: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::F64: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.eq\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#267-281\">source</a><a href=\"#method.eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html#tymethod.eq\" class=\"fn\">eq</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Self</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>self</code> and <code>other</code> values to be equal, and is used by <code>==</code>.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ne\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#261\">source</a></span><a href=\"#method.ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html#method.ne\" class=\"fn\">ne</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>!=</code>. The default implementation is almost always sufficient,\nand should not be overridden without very good reason.</div></details></div></details>","PartialEq","numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"]]
};if (window.register_type_impls) {window.register_type_impls(type_impls);} else {window.pending_type_impls = type_impls;}})()