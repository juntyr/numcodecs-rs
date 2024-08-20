use std::any::Any;

use ndarray::{ArrayViewD, ArrayViewMutD, CowArray};
use numcodecs::{
    AnyArray, AnyArrayView, AnyArrayViewMut, AnyCowArray, Codec, DynCodec, DynCodecType,
};
use numpy::{
    IxDyn, PyArray, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    intern,
    prelude::*,
    types::{IntoPyDict, PyDict, PyString, PyType},
    PyTypeInfo,
};
use pythonize::{pythonize, Depythonizer, Pythonizer};

use crate::{PyCodec, PyCodecClass, PyCodecClassAdapter, PyCodecRegistry};

/// Export the [`DynCodecType`] `ty` to Python by generating a fresh
/// [`PyCodecClass`] inside `module` and registering it with the
/// [`PyCodecRegistry`].
///
/// # Errors
///
/// Errors if generating or exporting the fresh [`PyCodecClass`] fails.
pub fn export_codec_class<'py, T: DynCodecType>(
    py: Python<'py>,
    ty: T,
    module: Borrowed<'_, 'py, PyModule>,
) -> Result<Bound<'py, PyCodecClass>, PyErr> {
    let codec_id = String::from(ty.codec_id());
    let codec_class_name = convert_case::Casing::to_case(&codec_id, convert_case::Case::Pascal);

    let codec_class: Bound<PyCodecClass> =
        // re-exporting a Python codec class should roundtrip
        if let Some(adapter) = (&ty as &dyn Any).downcast_ref::<PyCodecClassAdapter>() {
            adapter.as_codec_class(py).clone()
        } else {
            let codec_config_schema = pythonize(py, &ty.codec_config_schema())?;

            let codec_class_bases = (
                RustCodec::type_object_bound(py),
                PyCodec::type_object_bound(py),
            );

            let codec_class_namespace = [
                (intern!(py, "__module__"), module.name()?.into_any()),
                // (
                //     intern!(py, "__doc__"),
                //     PyString::new_bound(py, &documentation).into_any(),
                // ),
                (
                    intern!(py, "_ty"),
                    Bound::new(py, RustCodecType { ty: Box::new(ty) })?.into_any(),
                ),
                (
                    intern!(py, "codec_id"),
                    PyString::new_bound(py, &codec_id).into_any(),
                ),
                (
                    intern!(py, "__schema__"),
                    codec_config_schema.into_bound(py),
                ),
                // (
                //     intern!(py, "__init__"),
                //     py.eval_bound(&format!("lambda self, {signature}: None"), None, None)?,
                // ),
            ]
            .into_py_dict_bound(py);

            PyType::type_object_bound(py)
                .call1((&codec_class_name, codec_class_bases, codec_class_namespace))?
                .extract()?
        };

    module.add(codec_class_name.as_str(), &codec_class)?;

    PyCodecRegistry::register_codec(codec_class.as_borrowed(), None)?;

    Ok(codec_class)
}

#[pyclass(frozen)]
struct RustCodecType {
    ty: Box<dyn 'static + Send + Sync + AnyCodecType>,
}

trait AnyCodec {
    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, PyErr>;

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, PyErr>;

    fn decode_into(&self, encoded: AnyArrayView, decoded: AnyArrayViewMut) -> Result<(), PyErr>;

    fn get_config<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr>;
}

impl<T: DynCodec> AnyCodec for T {
    fn encode(&self, data: AnyCowArray) -> Result<AnyArray, PyErr> {
        <T as Codec>::encode(self, data).map_err(|err| PyRuntimeError::new_err(format!("{err}")))
    }

    fn decode(&self, encoded: AnyCowArray) -> Result<AnyArray, PyErr> {
        <T as Codec>::decode(self, encoded).map_err(|err| PyRuntimeError::new_err(format!("{err}")))
    }

    fn decode_into(&self, encoded: AnyArrayView, decoded: AnyArrayViewMut) -> Result<(), PyErr> {
        <T as Codec>::decode_into(self, encoded, decoded)
            .map_err(|err| PyRuntimeError::new_err(format!("{err}")))
    }

    fn get_config<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr> {
        <T as DynCodec>::get_config(self, Pythonizer::new(py))?.extract(py)
    }
}

trait AnyCodecType {
    fn codec_from_config<'py>(
        &self,
        config: Bound<'py, PyDict>,
    ) -> Result<Box<dyn 'static + Send + Sync + AnyCodec>, PyErr>;
}

impl<T: DynCodecType> AnyCodecType for T {
    fn codec_from_config<'py>(
        &self,
        config: Bound<'py, PyDict>,
    ) -> Result<Box<dyn 'static + Send + Sync + AnyCodec>, PyErr> {
        match <T as DynCodecType>::codec_from_config(
            self,
            &mut Depythonizer::from_object_bound(config.into_any()),
        ) {
            Ok(codec) => Ok(Box::new(codec)),
            Err(err) => Err(err.into()),
        }
    }
}

#[pyclass(subclass, frozen)]
struct RustCodec {
    cls_module: String,
    cls_name: String,
    codec: Box<dyn 'static + Send + Sync + AnyCodec>,
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
            .getattr(intern!(py, "_ty"))
            .map_err(|_| {
                PyValueError::new_err(format!(
                    "{cls_module}.{cls_name} is not linked to a Rust codec type"
                ))
            })?
            .extract()?;
        let ty: PyRef<RustCodecType> = ty.try_borrow()?;

        let codec = ty
            .ty
            .codec_from_config(kwargs.unwrap_or_else(|| PyDict::new_bound(py)))?;

        Ok(Self {
            cls_module,
            cls_name,
            codec,
        })
    }

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

    fn get_config<'py>(&self, py: Python<'py>) -> Result<Bound<'py, PyDict>, PyErr> {
        self.codec.get_config(py)
    }

    #[classmethod]
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
        let py_this: Py<PyAny> = this.into_py(py);

        let mut repr = py_this.bind(py).get_type().name()?.into_owned();
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
        process: impl FnOnce(
            &(dyn 'static + Send + Sync + AnyCodec),
            AnyCowArray,
        ) -> Result<AnyArray, PyErr>,
        class_method: &str,
    ) -> Result<Bound<'py, PyAny>, PyErr> {
        Self::with_pyarraylike_as_cow(py, buf, class_method, |data| {
            let processed = process(&*self.codec, data)?;
            Self::any_array_into_pyarray(py, processed, class_method)
        })
    }

    fn process_into<'py>(
        &self,
        py: Python<'py>,
        buf: Borrowed<'_, 'py, PyAny>,
        out: Borrowed<'_, 'py, PyAny>,
        process: impl FnOnce(
            &(dyn 'static + Send + Sync + AnyCodec),
            AnyArrayView,
            AnyArrayViewMut,
        ) -> Result<(), PyErr>,
        class_method: &str,
    ) -> Result<(), PyErr> {
        Self::with_pyarraylike_as_view(py, buf, class_method, |data| {
            Self::with_pyarraylike_as_view_mut(py, out, class_method, |data_out| {
                process(&*self.codec, data, data_out)
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

        let as_array = py
            .import_bound(intern!(py, "numpy"))?
            .getattr(intern!(py, "asarray"))?;

        let data: Bound<PyUntypedArray> = as_array.call1((buf,))?.extract()?;
        let dtype = data.dtype();

        if dtype.is_equiv_to(&numpy::dtype_bound::<u8>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u8>>()?.into(), |a| {
                with(AnyCowArray::U8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u16>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u16>>()?.into(), |a| {
                with(AnyCowArray::U16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u32>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u32>>()?.into(), |a| {
                with(AnyCowArray::U32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u64>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<u64>>()?.into(), |a| {
                with(AnyCowArray::U64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i8>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i8>>()?.into(), |a| {
                with(AnyCowArray::I8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i16>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i16>>()?.into(), |a| {
                with(AnyCowArray::I16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i32>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i32>>()?.into(), |a| {
                with(AnyCowArray::I32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i64>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<i64>>()?.into(), |a| {
                with(AnyCowArray::I64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<f32>(py)) {
            with_pyarraylike_as_cow_inner(data.downcast::<PyArrayDyn<f32>>()?.into(), |a| {
                with(AnyCowArray::F32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<f64>(py)) {
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

        let as_array = py
            .import_bound(intern!(py, "numpy"))?
            .getattr(intern!(py, "asarray"))?;

        let data: Bound<PyUntypedArray> = as_array.call1((buf,))?.extract()?;
        let dtype = data.dtype();

        if dtype.is_equiv_to(&numpy::dtype_bound::<u8>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u8>>()?.into(), |a| {
                with(AnyArrayView::U8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u16>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u16>>()?.into(), |a| {
                with(AnyArrayView::U16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u32>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u32>>()?.into(), |a| {
                with(AnyArrayView::U32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u64>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<u64>>()?.into(), |a| {
                with(AnyArrayView::U64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i8>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i8>>()?.into(), |a| {
                with(AnyArrayView::I8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i16>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i16>>()?.into(), |a| {
                with(AnyArrayView::I16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i32>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i32>>()?.into(), |a| {
                with(AnyArrayView::I32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i64>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<i64>>()?.into(), |a| {
                with(AnyArrayView::I64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<f32>(py)) {
            with_pyarraylike_as_view_inner(data.downcast::<PyArrayDyn<f32>>()?.into(), |a| {
                with(AnyArrayView::F32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<f64>(py)) {
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

        let as_array = py
            .import_bound(intern!(py, "numpy"))?
            .getattr(intern!(py, "asarray"))?;

        let data: Bound<PyUntypedArray> = as_array.call1((buf,))?.extract()?;
        let dtype = data.dtype();

        if dtype.is_equiv_to(&numpy::dtype_bound::<u8>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u8>>()?.into(), |a| {
                with(AnyArrayViewMut::U8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u16>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u16>>()?.into(), |a| {
                with(AnyArrayViewMut::U16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u32>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u32>>()?.into(), |a| {
                with(AnyArrayViewMut::U32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<u64>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<u64>>()?.into(), |a| {
                with(AnyArrayViewMut::U64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i8>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i8>>()?.into(), |a| {
                with(AnyArrayViewMut::I8(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i16>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i16>>()?.into(), |a| {
                with(AnyArrayViewMut::I16(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i32>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i32>>()?.into(), |a| {
                with(AnyArrayViewMut::I32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<i64>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<i64>>()?.into(), |a| {
                with(AnyArrayViewMut::I64(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<f32>(py)) {
            with_pyarraylike_as_view_mut_inner(data.downcast::<PyArrayDyn<f32>>()?.into(), |a| {
                with(AnyArrayViewMut::F32(a))
            })
        } else if dtype.is_equiv_to(&numpy::dtype_bound::<f64>(py)) {
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
            AnyArray::U8(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::U16(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::U32(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::U64(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::I8(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::I16(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::I32(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::I64(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::F32(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            AnyArray::F64(a) => Ok(PyArray::from_owned_array_bound(py, a).into_any()),
            array => Err(PyTypeError::new_err(format!(
                "{class_method} returned unsupported dtype `{}`",
                array.dtype(),
            ))),
        }
    }
}
