use std::{any::Any, ffi::CString, mem::ManuallyDrop};

use ndarray::{ArrayViewD, ArrayViewMutD, CowArray};
use numcodecs::{
    AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec, DynCodecType,
};
use numpy::{
    IxDyn, PyArray, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::PyTypeError,
    intern,
    marker::Ungil,
    prelude::*,
    types::{IntoPyDict, PyDict, PyString, PyType},
    PyTypeInfo,
};
use pyo3_error::PyErrChain;
use pythonize::{pythonize, Depythonizer};

use crate::{
    schema::{docs_from_schema, signature_from_schema},
    utils::numpy_asarray,
    PyCodec, PyCodecClass, PyCodecClassAdapter, PyCodecRegistry,
};

/// Export the [`DynCodecType`] `ty` to Python by generating a fresh
/// [`PyCodecClass`] inside `module` and registering it with the
/// [`PyCodecRegistry`].
///
/// # Errors
///
/// Errors if generating or exporting the fresh [`PyCodecClass`] fails.
pub fn export_codec_class<'py, T: DynCodecType<Codec: Ungil> + Ungil>(
    py: Python<'py>,
    ty: T,
    module: Borrowed<'_, 'py, PyModule>,
) -> Result<Bound<'py, PyCodecClass>, PyErr> {
    let codec_id = String::from(ty.codec_id());

    // special case for codec ids ending in .rs (we're writing Rust codecs after all)
    let codec_id_no_rs = codec_id.strip_suffix(".rs").unwrap_or(&codec_id);
    // derive the codec name, without any prefix
    let codec_name = match codec_id_no_rs.rsplit_once('.') {
        Some((_prefix, name)) => name,
        None => codec_id_no_rs,
    };
    let codec_class_name = convert_case::Casing::to_case(&codec_name, convert_case::Case::Pascal);

    let codec_class: Bound<PyCodecClass> =
        // re-exporting a Python codec class should roundtrip
        if let Some(adapter) = (&ty as &dyn Any).downcast_ref::<PyCodecClassAdapter>() {
            adapter.as_codec_class(py).clone()
        } else {
            let codec_config_schema = ty.codec_config_schema();

            let codec_class_bases = (
                RustCodec::type_object(py),
                PyCodec::type_object(py),
            );

            let codec_ty = RustCodecType { ty: ManuallyDrop::new(Box::new(ty)) };

            let codec_class_namespace = [
                (intern!(py, "__module__"), module.name()?.into_any()),
                (
                    intern!(py, "__doc__"),
                    docs_from_schema(&codec_config_schema).into_pyobject(py)?,
                ),
                (
                    intern!(py, RustCodec::TYPE_ATTRIBUTE),
                    Bound::new(py, codec_ty)?.into_any(),
                ),
                (
                    intern!(py, "codec_id"),
                    PyString::new(py, &codec_id).into_any(),
                ),
                (
                    intern!(py, RustCodec::SCHEMA_ATTRIBUTE),
                    pythonize(py, &codec_config_schema)?,
                ),
                (
                    intern!(py, "__init__"),
                    py.eval(&CString::new(format!(
                        "lambda {}: None",
                        signature_from_schema(&codec_config_schema),
                    ))?, None, None)?,
                ),
            ]
            .into_py_dict(py)?;

            PyType::type_object(py)
                .call1((&codec_class_name, codec_class_bases, codec_class_namespace))?
                .extract()?
        };

    module.add(codec_class_name.as_str(), &codec_class)?;

    PyCodecRegistry::register_codec(codec_class.as_borrowed(), None)?;

    Ok(codec_class)
}

#[expect(clippy::redundant_pub_crate)]
#[pyclass(frozen, module = "numcodecs._rust", name = "_RustCodecType")]
/// Rust-implemented codec type container.
pub(crate) struct RustCodecType {
    ty: ManuallyDrop<Box<dyn AnyCodecType>>,
}

impl Drop for RustCodecType {
    fn drop(&mut self) {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                #[allow(unsafe_code)]
                unsafe {
                    ManuallyDrop::drop(&mut self.ty);
                }
            });
        });
    }
}

impl RustCodecType {
    pub fn downcast<T: DynCodecType>(&self) -> Option<&T> {
        self.ty.as_any().downcast_ref()
    }
}

trait AnyCodec: 'static + Send + Sync + Ungil {
    fn encode(&self, py: Python, data: AnyCowArray) -> Result<AnyArray, PyErr>;

    fn decode(&self, py: Python, encoded: AnyCowArray) -> Result<AnyArray, PyErr>;

    fn decode_into(
        &self,
        py: Python,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), PyErr>;

    fn get_config<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr>;

    fn as_any(&self) -> &dyn Any;
}

impl<T: DynCodec + Ungil> AnyCodec for T {
    fn encode(&self, py: Python, data: AnyCowArray) -> Result<AnyArray, PyErr> {
        py.allow_threads(|| <T as Codec>::encode(self, data))
            .map_err(|err| PyErrChain::pyerr_from_err(py, err))
    }

    fn decode(&self, py: Python, encoded: AnyCowArray) -> Result<AnyArray, PyErr> {
        py.allow_threads(|| <T as Codec>::decode(self, encoded))
            .map_err(|err| PyErrChain::pyerr_from_err(py, err))
    }

    fn decode_into(
        &self,
        py: Python,
        encoded: AnyArrayView,
        decoded: AnyArrayViewMut,
    ) -> Result<(), PyErr> {
        py.allow_threads(|| <T as Codec>::decode_into(self, encoded, decoded))
            .map_err(|err| PyErrChain::pyerr_from_err(py, err))
    }

    fn get_config<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr> {
        let config: serde_json::Value = py
            .allow_threads(|| <T as DynCodec>::get_config(self, serde_json::value::Serializer))
            .map_err(|err| PyErrChain::pyerr_from_err(py, err))?;
        pythonize::pythonize(py, &config)?.extract()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

trait AnyCodecType: 'static + Send + Sync + Ungil {
    fn codec_from_config<'py>(
        &self,
        py: Python<'py>,
        cls_module: String,
        cls_name: String,
        config: Bound<'py, PyDict>,
    ) -> Result<RustCodec, PyErr>;

    fn as_any(&self) -> &dyn Any;
}

impl<T: DynCodecType + Ungil> AnyCodecType for T {
    fn codec_from_config<'py>(
        &self,
        py: Python<'py>,
        cls_module: String,
        cls_name: String,
        config: Bound<'py, PyDict>,
    ) -> Result<RustCodec, PyErr> {
        let config = serde_transcode::transcode(
            &mut Depythonizer::from_object(config.as_any()),
            serde_json::value::Serializer,
        )
        .map_err(|err| PyErrChain::pyerr_from_err(py, err))?;

        py.allow_threads(|| -> Result<RustCodec, serde_json::Error> {
            let codec = <T as DynCodecType>::codec_from_config(self, config)?;

            Ok(RustCodec {
                cls_module,
                cls_name,
                codec: ManuallyDrop::new(Box::new(codec)),
            })
        })
        .map_err(|err| PyErrChain::pyerr_from_err(py, err))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[expect(clippy::redundant_pub_crate)]
#[pyclass(subclass, frozen, module = "numcodecs._rust")]
/// Rust-implemented codec abstract base class.
///
/// This class implements the [`numcodecs.abc.Codec`][numcodecs.abc.Codec] API.
pub(crate) struct RustCodec {
    cls_module: String,
    cls_name: String,
    codec: ManuallyDrop<Box<dyn AnyCodec>>,
}

impl Drop for RustCodec {
    fn drop(&mut self) {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                #[allow(unsafe_code)]
                unsafe {
                    ManuallyDrop::drop(&mut self.codec);
                }
            });
        });
    }
}

impl RustCodec {
    pub const SCHEMA_ATTRIBUTE: &'static str = "__schema__";
    pub const TYPE_ATTRIBUTE: &'static str = "_ty";

    pub fn downcast<T: DynCodec>(&self) -> Option<&T> {
        self.codec.as_any().downcast_ref()
    }
}

#[pymethods]
impl RustCodec {
    #[new]
    #[classmethod]
    #[pyo3(signature = (**kwargs))]
    fn new<'py>(
        cls: &Bound<'py, PyType>,
        py: Python<'py>,
        kwargs: Option<Bound<'py, PyDict>>,
    ) -> Result<Self, PyErr> {
        let cls: &Bound<PyCodecClass> = cls.downcast()?;
        let cls_module: String = cls.getattr(intern!(py, "__module__"))?.extract()?;
        let cls_name: String = cls.getattr(intern!(py, "__name__"))?.extract()?;

        let ty: Bound<RustCodecType> = cls
            .getattr(intern!(py, RustCodec::TYPE_ATTRIBUTE))
            .map_err(|_| {
                PyTypeError::new_err(format!(
                    "{cls_module}.{cls_name} is not linked to a Rust codec type"
                ))
            })?
            .extract()?;
        let ty: PyRef<RustCodecType> = ty.try_borrow()?;

        ty.ty.codec_from_config(
            py,
            cls_module,
            cls_name,
            kwargs.unwrap_or_else(|| PyDict::new(py)),
        )
    }

    /// Encode the data in `buf`.
    ///
    /// Parameters
    /// ----------
    /// buf : Buffer
    ///     Data to be encoded. May be any object supporting the new-style
    ///     buffer protocol.
    ///
    /// Returns
    /// -------
    /// enc : Buffer
    ///     Encoded data. May be any object supporting the new-style buffer
    ///     protocol.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        buf: &Bound<'py, PyAny>,
    ) -> Result<Bound<'py, PyAny>, PyErr> {
        self.process(
            py,
            buf.as_borrowed(),
            AnyCodec::encode,
            &format!("{}.{}::encode", self.cls_module, self.cls_name),
        )
    }

    #[pyo3(signature = (buf, out=None))]
    /// Decode the data in `buf`.
    ///
    /// Parameters
    /// ----------
    /// buf : Buffer
    ///     Encoded data. May be any object supporting the new-style buffer
    ///     protocol.
    /// out : Buffer, optional
    ///     Writeable buffer to store decoded data. N.B. if provided, this buffer must
    ///     be exactly the right size to store the decoded data.
    ///
    /// Returns
    /// -------
    /// dec : Buffer
    ///     Decoded data. May be any object supporting the new-style
    ///     buffer protocol.
    fn decode<'py>(
        &self,
        py: Python<'py>,
        buf: &Bound<'py, PyAny>,
        out: Option<Bound<'py, PyAny>>,
    ) -> Result<Bound<'py, PyAny>, PyErr> {
        let class_method = &format!("{}.{}::decode", self.cls_module, self.cls_name);
        if let Some(out) = out {
            self.process_into(
                py,
                buf.as_borrowed(),
                out.as_borrowed(),
                AnyCodec::decode_into,
                class_method,
            )?;
            Ok(out)
        } else {
            self.process(py, buf.as_borrowed(), AnyCodec::decode, class_method)
        }
    }

    /// Returns the configuration of the codec.
    ///
    /// [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
    /// can be used to reconstruct this codec from the returned config.
    ///
    /// Returns
    /// -------
    /// config : dict
    ///     Configuration of the codec.
    fn get_config<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr> {
        self.codec.get_config(py)
    }

    #[classmethod]
    /// Instantiate the codec from a configuration [`dict`][dict].
    ///
    /// Parameters
    /// ----------
    /// config : dict
    ///     Configuration of the codec.
    ///
    /// Returns
    /// -------
    /// codec : Self
    ///     Instantiated codec.
    fn from_config<'py>(
        cls: &Bound<'py, PyType>,
        config: &Bound<'py, PyDict>,
    ) -> Result<Bound<'py, PyCodec>, PyErr> {
        let cls: Bound<PyCodecClass> = cls.extract()?;

        // Ensures that cls(**config) is called and an instance of cls is returned
        cls.call((), Some(config))?.extract()
    }

    fn __repr__(this: PyRef<Self>, py: Python) -> Result<String, PyErr> {
        let config = this.get_config(py)?;
        let Ok(py_this) = this.into_pyobject(py);

        let mut repr = py_this.get_type().name()?.to_cow()?.into_owned();
        repr.push('(');

        let mut first = true;

        for (name, value) in config.iter() {
            let name: String = name.extract()?;

            if name == "id" {
                // Exclude the id config parameter from the repr
                continue;
            }

            let value_repr: String = value.repr()?.extract()?;

            if !first {
                repr.push_str(", ");
            }
            first = false;

            repr.push_str(&name);
            repr.push('=');
            repr.push_str(&value_repr);
        }

        repr.push(')');

        Ok(repr)
    }
}

impl RustCodec {
    fn process<'py>(
        &self,
        py: Python<'py>,
        buf: Borrowed<'_, 'py, PyAny>,
        process: impl FnOnce(&dyn AnyCodec, Python, AnyCowArray) -> Result<AnyArray, PyErr>,
        class_method: &str,
    ) -> Result<Bound<'py, PyAny>, PyErr> {
        Self::with_pyarraylike_as_cow(py, buf, class_method, |data| {
            let processed = process(&**self.codec, py, data)?;
            Self::any_array_into_pyarray(py, processed, class_method)
        })
    }

    fn process_into<'py>(
        &self,
        py: Python<'py>,
        buf: Borrowed<'_, 'py, PyAny>,
        out: Borrowed<'_, 'py, PyAny>,
        process: impl FnOnce(&dyn AnyCodec, Python, AnyArrayView, AnyArrayViewMut) -> Result<(), PyErr>,
        class_method: &str,
    ) -> Result<(), PyErr> {
        Self::with_pyarraylike_as_view(py, buf, class_method, |data| {
            Self::with_pyarraylike_as_view_mut(py, out, class_method, |data_out| {
                process(&**self.codec, py, data, data_out)
            })
        })
    }

    fn with_pyarraylike_as_cow<'py, O>(
        py: Python<'py>,
        buf: Borrowed<'_, 'py, PyAny>,
        class_method: &str,
        with: impl for<'a> FnOnce(AnyCowArray<'a>) -> Result<O, PyErr>,
    ) -> Result<O, PyErr> {
        fn with_pyarraylike_as_cow_inner<T: numpy::Element, O>(
            data: Borrowed<PyArrayDyn<T>>,
            with: impl for<'a> FnOnce(CowArray<'a, T, IxDyn>) -> Result<O, PyErr>,
        ) -> Result<O, PyErr> {
            let readonly_data = data.try_readonly()?;
            with(readonly_data.as_array().into())
        }

        let data = numpy_asarray(py, buf)?;
        let dtype = data.dtype();

        if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u8>>()?.into(), |a| {
                with(AnyCowArray::U8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u16>>()?.into(), |a| {
                with(AnyCowArray::U16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u32>>()?.into(), |a| {
                with(AnyCowArray::U32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u64>>()?.into(), |a| {
                with(AnyCowArray::U64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i8>>()?.into(), |a| {
                with(AnyCowArray::I8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i16>>()?.into(), |a| {
                with(AnyCowArray::I16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i32>>()?.into(), |a| {
                with(AnyCowArray::I32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i64>>()?.into(), |a| {
                with(AnyCowArray::I64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<f32>>()?.into(), |a| {
                with(AnyCowArray::F32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<f64>>()?.into(), |a| {
                with(AnyCowArray::F64(a))
            })
        } else {
            Err(PyTypeError::new_err(format!(
                "{class_method} received buffer of unsupported dtype `{dtype}`",
            )))
        }
    }

    fn with_pyarraylike_as_view<'py, O>(
        py: Python<'py>,
        buf: Borrowed<'_, 'py, PyAny>,
        class_method: &str,
        with: impl for<'a> FnOnce(AnyArrayView<'a>) -> Result<O, PyErr>,
    ) -> Result<O, PyErr> {
        fn with_pyarraylike_as_view_inner<T: numpy::Element, O>(
            data: Borrowed<PyArrayDyn<T>>,
            with: impl for<'a> FnOnce(ArrayViewD<'a, T>) -> Result<O, PyErr>,
        ) -> Result<O, PyErr> {
            let readonly_data = data.try_readonly()?;
            with(readonly_data.as_array())
        }

        let data = numpy_asarray(py, buf)?;
        let dtype = data.dtype();

        if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u8>>()?.into(), |a| {
                with(AnyArrayView::U8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u16>>()?.into(), |a| {
                with(AnyArrayView::U16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u32>>()?.into(), |a| {
                with(AnyArrayView::U32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u64>>()?.into(), |a| {
                with(AnyArrayView::U64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i8>>()?.into(), |a| {
                with(AnyArrayView::I8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i16>>()?.into(), |a| {
                with(AnyArrayView::I16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i32>>()?.into(), |a| {
                with(AnyArrayView::I32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i64>>()?.into(), |a| {
                with(AnyArrayView::I64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<f32>>()?.into(), |a| {
                with(AnyArrayView::F32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<f64>>()?.into(), |a| {
                with(AnyArrayView::F64(a))
            })
        } else {
            Err(PyTypeError::new_err(format!(
                "{class_method} received buffer of unsupported dtype `{dtype}`",
            )))
        }
    }

    fn with_pyarraylike_as_view_mut<'py, O>(
        py: Python<'py>,
        buf: Borrowed<'_, 'py, PyAny>,
        class_method: &str,
        with: impl for<'a> FnOnce(AnyArrayViewMut<'a>) -> Result<O, PyErr>,
    ) -> Result<O, PyErr> {
        fn with_pyarraylike_as_view_mut_inner<T: numpy::Element, O>(
            data: Borrowed<PyArrayDyn<T>>,
            with: impl for<'a> FnOnce(ArrayViewMutD<'a, T>) -> Result<O, PyErr>,
        ) -> Result<O, PyErr> {
            let mut readwrite_data = data.try_readwrite()?;
            with(readwrite_data.as_array_mut())
        }

        let data = numpy_asarray(py, buf)?;
        let dtype = data.dtype();

        if dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u8>>()?.into(), |a| {
                with(AnyArrayViewMut::U8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u16>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u16>>()?.into(), |a| {
                with(AnyArrayViewMut::U16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u32>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u32>>()?.into(), |a| {
                with(AnyArrayViewMut::U32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<u64>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u64>>()?.into(), |a| {
                with(AnyArrayViewMut::U64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i8>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i8>>()?.into(), |a| {
                with(AnyArrayViewMut::I8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i16>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i16>>()?.into(), |a| {
                with(AnyArrayViewMut::I16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i32>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i32>>()?.into(), |a| {
                with(AnyArrayViewMut::I32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<i64>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i64>>()?.into(), |a| {
                with(AnyArrayViewMut::I64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<f32>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<f32>>()?.into(), |a| {
                with(AnyArrayViewMut::F32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype::<f64>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<f64>>()?.into(), |a| {
                with(AnyArrayViewMut::F64(a))
            })
        } else {
            Err(PyTypeError::new_err(format!(
                "{class_method} received buffer of unsupported dtype `{dtype}`",
            )))
        }
    }

    fn any_array_into_pyarray<'py>(
        py: Python<'py>,
        array: AnyArray,
        class_method: &str,
    ) -> Result<Bound<'py, PyAny>, PyErr> {
        match array {
            AnyArray::U8(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::U16(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::U32(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::U64(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::I8(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::I16(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::I32(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::I64(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::F32(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            AnyArray::F64(a) => Ok(PyArray::from_owned_array(py, a).into_any()),
            array => Err(PyTypeError::new_err(format!(
                "{class_method} returned unsupported dtype `{}`",
                array.dtype(),
            ))),
        }
    }
}
