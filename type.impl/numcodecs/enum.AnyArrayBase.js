(function() {
    var type_impls = Object.fromEntries([["numcodecs",[["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#164-281\">source</a><a href=\"#impl-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.view\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#179-192\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.view\" class=\"fn\">view</a>(&amp;self) -&gt; <a class=\"type\" href=\"numcodecs/type.AnyArrayView.html\" title=\"type numcodecs::AnyArrayView\">AnyArrayView</a>&lt;'_&gt;</h4></section></summary><div class=\"docblock\"><p>Returns a read-only view of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.cow\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#196-209\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.cow\" class=\"fn\">cow</a>(&amp;self) -&gt; <a class=\"type\" href=\"numcodecs/type.AnyCowArray.html\" title=\"type numcodecs::AnyCowArray\">AnyCowArray</a>&lt;'_&gt;</h4></section></summary><div class=\"docblock\"><p>Returns a copy-on-write view of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.into_owned\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#214-227\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.into_owned\" class=\"fn\">into_owned</a>(self) -&gt; <a class=\"type\" href=\"numcodecs/type.AnyArray.html\" title=\"type numcodecs::AnyArray\">AnyArray</a></h4></section></summary><div class=\"docblock\"><p>Turns the array into a uniquely owned array, cloning the array elements\nif necessary.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.as_bytes\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#237-280\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.as_bytes\" class=\"fn\">as_bytes</a>(&amp;self) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/alloc/borrow/enum.Cow.html\" title=\"enum alloc::borrow::Cow\">Cow</a>&lt;'_, [<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u8.html\">u8</a>]&gt;</h4></section></summary><div class=\"docblock\"><p>Returns the array’s data as a byte slice.</p>\n<p>If the array is contiguous and in standard order, i.e. if the element\norder in memory corresponds to the logical order of the array’s\nelements, a view of the data is returned without cloning.</p>\n<p>Otherwise, the data is cloned and put into standard order first.</p>\n</div></details></div></details>",0,"numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#283-434\">source</a><a href=\"#impl-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.DataMut.html\" title=\"trait ndarray::data_traits::DataMut\">DataMut</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.view_mut\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#298-311\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.view_mut\" class=\"fn\">view_mut</a>(&amp;mut self) -&gt; <a class=\"type\" href=\"numcodecs/type.AnyArrayViewMut.html\" title=\"type numcodecs::AnyArrayViewMut\">AnyArrayViewMut</a>&lt;'_&gt;</h4></section></summary><div class=\"docblock\"><p>Returns a read-write view of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.with_bytes_mut\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#322-369\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.with_bytes_mut\" class=\"fn\">with_bytes_mut</a>&lt;O&gt;(&amp;mut self, with: impl <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/ops/function/trait.FnOnce.html\" title=\"trait core::ops::function::FnOnce\">FnOnce</a>(&amp;mut [<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.u8.html\">u8</a>]) -&gt; O) -&gt; O</h4></section></summary><div class=\"docblock\"><p>Provides access to the array’s data as a mutable byte slice.</p>\n<p>If the array is contiguous and in standard order, i.e. if the element\norder in memory corresponds to the logical order of the array’s\nelements, a mutable view of the data is returned without cloning.</p>\n<p>Otherwise, the data is cloned and put into standard order first, and\nlater copied back into the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.assign\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#380-433\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.assign\" class=\"fn\">assign</a>&lt;U: <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt;(\n    &amp;mut self,\n    src: &amp;<a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;U&gt;,\n) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/result/enum.Result.html\" title=\"enum core::result::Result\">Result</a>&lt;<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.unit.html\">()</a>, <a class=\"enum\" href=\"numcodecs/enum.AnyArrayAssignError.html\" title=\"enum numcodecs::AnyArrayAssignError\">AnyArrayAssignError</a>&gt;<div class=\"where\">where\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    U::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div></h4></section></summary><div class=\"docblock\"><p>Perform an elementwise assigment to <code>self</code> from <code>src</code>.</p>\n<h5 id=\"errors\"><a class=\"doc-anchor\" href=\"#errors\">§</a>Errors</h5>\n<p>Errors with</p>\n<ul>\n<li><a href=\"numcodecs/enum.AnyArrayAssignError.html#variant.DTypeMismatch\" title=\"variant numcodecs::AnyArrayAssignError::DTypeMismatch\"><code>AnyArrayAssignError::DTypeMismatch</code></a> if the dtypes of <code>self</code> and\n<code>src</code> do not match</li>\n<li><a href=\"numcodecs/enum.AnyArrayAssignError.html#variant.ShapeMismatch\" title=\"variant numcodecs::AnyArrayAssignError::ShapeMismatch\"><code>AnyArrayAssignError::ShapeMismatch</code></a> if the shapes of <code>self</code> and\n<code>src</code> do not match</li>\n</ul>\n</div></details></div></details>",0,"numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#38-162\">source</a><a href=\"#impl-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;</h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.len\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#40-53\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.len\" class=\"fn\">len</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a></h4></section></summary><div class=\"docblock\"><p>Returns the total number of elements in the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.is_empty\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#56-69\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.is_empty\" class=\"fn\">is_empty</a>(&amp;self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class=\"docblock\"><p>Returns whether the array has any elements.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.dtype\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#72-85\">source</a><h4 class=\"code-header\">pub const fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.dtype\" class=\"fn\">dtype</a>(&amp;self) -&gt; <a class=\"enum\" href=\"numcodecs/enum.AnyArrayDType.html\" title=\"enum numcodecs::AnyArrayDType\">AnyArrayDType</a></h4></section></summary><div class=\"docblock\"><p>Returns the dtype of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.shape\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#88-101\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.shape\" class=\"fn\">shape</a>(&amp;self) -&gt; &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.usize.html\">usize</a>]</h4></section></summary><div class=\"docblock\"><p>Returns the shape of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.strides\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#104-117\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.strides\" class=\"fn\">strides</a>(&amp;self) -&gt; &amp;[<a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.isize.html\">isize</a>]</h4></section></summary><div class=\"docblock\"><p>Return the strides of the array.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.as_typed\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#122-139\">source</a><h4 class=\"code-header\">pub const fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.as_typed\" class=\"fn\">as_typed</a>&lt;U: <a class=\"trait\" href=\"numcodecs/trait.ArrayDType.html\" title=\"trait numcodecs::ArrayDType\">ArrayDType</a>&gt;(\n    &amp;self,\n) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;<a class=\"struct\" href=\"https://docs.rs/ndarray/0.15/ndarray/struct.ArrayBase.html\" title=\"struct ndarray::ArrayBase\">ArrayBase</a>&lt;U::<a class=\"associatedtype\" href=\"numcodecs/trait.ArrayDType.html#associatedtype.RawData\" title=\"type numcodecs::ArrayDType::RawData\">RawData</a>&lt;T&gt;, <a class=\"type\" href=\"https://docs.rs/ndarray/0.15/ndarray/aliases/type.IxDyn.html\" title=\"type ndarray::aliases::IxDyn\">IxDyn</a>&gt;&gt;</h4></section></summary><div class=\"docblock\"><p>Returns the <code>U</code>-typed array in <code>Some(_)</code> iff the dtype of <code>U</code> matches\nthe dtype of this array. Returns <code>None</code> otherwise.</p>\n</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.as_typed_mut\" class=\"method\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#144-161\">source</a><h4 class=\"code-header\">pub fn <a href=\"numcodecs/enum.AnyArrayBase.html#tymethod.as_typed_mut\" class=\"fn\">as_typed_mut</a>&lt;U: <a class=\"trait\" href=\"numcodecs/trait.ArrayDType.html\" title=\"trait numcodecs::ArrayDType\">ArrayDType</a>&gt;(\n    &amp;mut self,\n) -&gt; <a class=\"enum\" href=\"https://doc.rust-lang.org/nightly/core/option/enum.Option.html\" title=\"enum core::option::Option\">Option</a>&lt;&amp;mut <a class=\"struct\" href=\"https://docs.rs/ndarray/0.15/ndarray/struct.ArrayBase.html\" title=\"struct ndarray::ArrayBase\">ArrayBase</a>&lt;U::<a class=\"associatedtype\" href=\"numcodecs/trait.ArrayDType.html#associatedtype.RawData\" title=\"type numcodecs::ArrayDType::RawData\">RawData</a>&lt;T&gt;, <a class=\"type\" href=\"https://docs.rs/ndarray/0.15/ndarray/aliases/type.IxDyn.html\" title=\"type ndarray::aliases::IxDyn\">IxDyn</a>&gt;&gt;</h4></section></summary><div class=\"docblock\"><p>Returns the <code>U</code>-typed array in <code>Some(_)</code> iff the dtype of <code>U</code> matches\nthe dtype of this array. Returns <code>None</code> otherwise.</p>\n</div></details></div></details>",0,"numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Clone-for-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#516-543\">source</a><a href=\"#impl-Clone-for-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html\" title=\"trait core::clone::Clone\">Clone</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.RawDataClone.html\" title=\"trait ndarray::data_traits::RawDataClone\">RawDataClone</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#529-542\">source</a><a href=\"#method.clone\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#tymethod.clone\" class=\"fn\">clone</a>(&amp;self) -&gt; Self</h4></section></summary><div class='docblock'>Returns a copy of the value. <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#tymethod.clone\">Read more</a></div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.clone_from\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/clone.rs.html#174\">source</a></span><a href=\"#method.clone_from\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#method.clone_from\" class=\"fn\">clone_from</a>(&amp;mut self, source: &amp;Self)</h4></section></summary><div class='docblock'>Performs copy-assignment from <code>source</code>. <a href=\"https://doc.rust-lang.org/nightly/core/clone/trait.Clone.html#method.clone_from\">Read more</a></div></details></div></details>","Clone","numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-Debug-for-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#545-572\">source</a><a href=\"#impl-Debug-for-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html\" title=\"trait core::fmt::Debug\">Debug</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.fmt\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#558-571\">source</a><a href=\"#method.fmt\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\" class=\"fn\">fmt</a>(&amp;self, fmt: &amp;mut <a class=\"struct\" href=\"https://doc.rust-lang.org/nightly/core/fmt/struct.Formatter.html\" title=\"struct core::fmt::Formatter\">Formatter</a>&lt;'_&gt;) -&gt; <a class=\"type\" href=\"https://doc.rust-lang.org/nightly/core/fmt/type.Result.html\" title=\"type core::fmt::Result\">Result</a></h4></section></summary><div class='docblock'>Formats the value using the given formatter. <a href=\"https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html#tymethod.fmt\">Read more</a></div></details></div></details>","Debug","numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"],["<details class=\"toggle implementors-toggle\" open><summary><section id=\"impl-PartialEq-for-AnyArrayBase%3CT%3E\" class=\"impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#574-602\">source</a><a href=\"#impl-PartialEq-for-AnyArrayBase%3CT%3E\" class=\"anchor\">§</a><h3 class=\"code-header\">impl&lt;T: <a class=\"trait\" href=\"numcodecs/trait.AnyRawData.html\" title=\"trait numcodecs::AnyRawData\">AnyRawData</a>&gt; <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html\" title=\"trait core::cmp::PartialEq\">PartialEq</a> for <a class=\"enum\" href=\"numcodecs/enum.AnyArrayBase.html\" title=\"enum numcodecs::AnyArrayBase\">AnyArrayBase</a>&lt;T&gt;<div class=\"where\">where\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U8\" title=\"type numcodecs::AnyRawData::U8\">U8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U16\" title=\"type numcodecs::AnyRawData::U16\">U16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U32\" title=\"type numcodecs::AnyRawData::U32\">U32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.U64\" title=\"type numcodecs::AnyRawData::U64\">U64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I8\" title=\"type numcodecs::AnyRawData::I8\">I8</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I16\" title=\"type numcodecs::AnyRawData::I16\">I16</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I32\" title=\"type numcodecs::AnyRawData::I32\">I32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.I64\" title=\"type numcodecs::AnyRawData::I64\">I64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F32\" title=\"type numcodecs::AnyRawData::F32\">F32</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,\n    T::<a class=\"associatedtype\" href=\"numcodecs/trait.AnyRawData.html#associatedtype.F64\" title=\"type numcodecs::AnyRawData::F64\">F64</a>: <a class=\"trait\" href=\"https://docs.rs/ndarray/0.15/ndarray/data_traits/trait.Data.html\" title=\"trait ndarray::data_traits::Data\">Data</a>,</div></h3></section></summary><div class=\"impl-items\"><details class=\"toggle method-toggle\" open><summary><section id=\"method.eq\" class=\"method trait-impl\"><a class=\"src rightside\" href=\"src/numcodecs/array.rs.html#587-601\">source</a><a href=\"#method.eq\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html#tymethod.eq\" class=\"fn\">eq</a>(&amp;self, other: &amp;Self) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>self</code> and <code>other</code> values to be equal, and is used by <code>==</code>.</div></details><details class=\"toggle method-toggle\" open><summary><section id=\"method.ne\" class=\"method trait-impl\"><span class=\"rightside\"><span class=\"since\" title=\"Stable since Rust version 1.0.0\">1.0.0</span> · <a class=\"src\" href=\"https://doc.rust-lang.org/nightly/src/core/cmp.rs.html#261\">source</a></span><a href=\"#method.ne\" class=\"anchor\">§</a><h4 class=\"code-header\">fn <a href=\"https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html#method.ne\" class=\"fn\">ne</a>(&amp;self, other: <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.reference.html\">&amp;Rhs</a>) -&gt; <a class=\"primitive\" href=\"https://doc.rust-lang.org/nightly/std/primitive.bool.html\">bool</a></h4></section></summary><div class='docblock'>Tests for <code>!=</code>. The default implementation is almost always sufficient,\nand should not be overridden without very good reason.</div></details></div></details>","PartialEq","numcodecs::array::AnyArcArray","numcodecs::array::AnyArray","numcodecs::array::AnyArrayView","numcodecs::array::AnyArrayViewMut","numcodecs::array::AnyCowArray"]]]]);
    if (window.register_type_impls) {
        window.register_type_impls(type_impls);
    } else {
        window.pending_type_impls = type_impls;
    }
})()
//{"start":55,"fragment_lengths":[38461]}