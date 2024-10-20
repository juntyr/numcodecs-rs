searchState.loadedDescShard("numcodecs", 0, "CI Status MSRV Latest Version Rust Doc Crate Rust Doc Main\nAn array where the data has shared ownership and is …\nAn array that owns its data uniquely.\nErrors that may occur when calling <code>AnyArrayBase::assign</code>.\nNumeric n-dimensional arrays with dynamic shapes.\nEnum of all dtypes included in <code>AnyArrayBase</code>.\nA read-only array view.\nA read-write array view.\nAn array with copy-on-write behavior.\nArray-representation support for all dtypes included in …\nTypes which are included in <code>AnyArrayDType</code>\nCodec identifier.\nCompression codec that <code>encode</code>s and <code>decode</code>s numeric …\nType of the instances of this codec type object.\nConfiguration type, from which the codec can be created …\n<code>AnyArrayDType</code> representation of this type\ncannot assign from mismatching <code>src</code> array to <code>dst</code>\nDynamically typed compression codec.\nType object for dynamically typed compression codecs.\nError type that may be returned during <code>encode</code>ing and <code>decode</code>…\nRepresentation for an <code>ArrayBase</code> containing this type\ncannot assign from array of shape <code>src</code> to one of shape <code>dst</code>\nStatically typed compression codec.\nUtility struct to serialize a <code>StaticCodec</code>’s …\nType object for statically typed compression codecs.\nType object type for this codec.\nReturns the array’s data as a byte slice.\nReturns the <code>U</code>-typed array in <code>Some(_)</code> iff the dtype of <code>U</code> …\nReturns the <code>U</code>-typed array in <code>Some(_)</code> iff the dtype of <code>U</code> …\nPerform an elementwise assignment to <code>self</code> from <code>src</code>.\nJSON schema for the codec’s configuration.\nInstantiate a codec of this type from a serialized <code>config</code>…\nUtility function to instantiate a codec of the given <code>ty</code>, …\nCodec identifier.\nThe configuration parameters\nReturns a copy-on-write view of the array.\nDecodes the <code>encoded</code> data and returns the result.\nDecodes the <code>encoded</code> data and writes the result into the …\nReturns the dtype of the array.\nEncodes the <code>data</code> and returns the result.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nReturns the argument unchanged.\nInstantiate a codec from its <code>config</code>uration.\nGet the configuration for this codec.\nSerializes the configuration parameters for this codec.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nCalls <code>U::from(self)</code>.\nReturns an owned copy-on-write array.\nReturns an owned copy-on-write array.\nTurns the array into a uniquely owned array, cloning the …\nReturns whether the array has any elements.\nReturns the total number of elements in the array.\nWraps the <code>config</code> so that it can be serialized together …\nStatically obtain the type for a statically typed codec.\nReturns the dtype of the type <code>T</code>\nUtility function to serialize a codec’s config together …\nReturns the shape of the array.\nReturns the size of the dtype in bytes.\nReturn the strides of the array.\nConverts the dtype to its (unsigned) binary equivalent.\nReturns the type object for this codec.\nReturns a read-only view of the array.\nReturns a read-write view of the array.\nProvides access to the array’s data as a mutable byte …\nCreate an array with zeros of <code>dtype</code> and shape <code>shape</code>, and …\nCreate an array with zeros of <code>dtype</code> and shape <code>shape</code>, and …\nCreate an array with zeros of <code>dtype</code> and shape <code>shape</code>.\nCreate an array with zeros of <code>dtype</code> and shape <code>shape</code>.\nDtype of the <code>dst</code> array into which the data is copied\nShape of the <code>dst</code> array into which the data is copied\nDtype of the <code>src</code> array from which the data is copied\nShape of the <code>src</code> array from which the data is copied")