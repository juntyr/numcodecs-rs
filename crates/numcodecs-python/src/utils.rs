use numpy::PyUntypedArray;
use pyo3::{prelude::*, sync::PyOnceLock};

pub fn numpy_asarray<'py>(
    py: Python<'py>,
    a: Borrowed<'_, 'py, PyAny>,
) -> Result<Bound<'py, PyUntypedArray>, PyErr> {
    static AS_ARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    let as_array = AS_ARRAY.import(py, "numpy", "asarray")?;

    as_array.call1((a,))?.extract()
}
