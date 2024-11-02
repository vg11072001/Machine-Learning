
- [ML Projects Open source guide](https://keymakr.com/blog/contribute-to-open-source-machine-learning-projects/)
- [7-open-source-machine-learning-projects-contribute-today](https://machinelearningmastery.com/7-open-source-machine-learning-projects-contribute-today/)
- [a-guide-to-open-source-for-ml-enthusiasts](https://dscunilag.medium.com/a-guide-to-open-source-for-ml-enthusiasts-a824d69a15b4#9a61)

# 1 .issue on TensorFlow repo
- `TensorFlow argmin function returns incorrect index when dealing with subnormal float values` [#77946](https://github.com/tensorflow/tensorflow/issues/77946)
- `argmax returns incorrect result for input containing Minimum number (TensorFlow 2.x)`  [#77853](https://github.com/tensorflow/tensorflow/issues/77853)

what i found till now issue starts with

- > https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/math_ops.py#L306
- >>   on local path [C:\Users\vanshika.gupta\AppData\Local\anaconda3\Lib\site-packages\tensorflow\python\ops\gen_math_ops.py]
- >>> ``TFE_Py_FastPathExecute `` [C:\Users\vanshika.gupta\AppData\Local\anaconda3\Lib\site-packages\tensorflow\python\_pywrap_tfe.pyi]
- >>>> 
```cpp
 m.def("TFE_Py_FastPathExecute", [](const py::args args) {
    // TFE_Py_FastPathExecute requires error checking prior to returning.
    return tensorflow::PyoOrThrow(TFE_Py_FastPathExecute_C(args.ptr()));
  });
```
https://github.com/tensorflow/tensorflow/blob/805ae2c26493334538f53244229cd573ba871711/tensorflow/python/tfe_wrapper.cc#L1278 
- >>>>
```cpp
PyObject* TFE_Py_FastPathExecute_C(PyObject* args) {
  tsl::profiler::TraceMe activity("TFE_Py_FastPathExecute_C",
                                  tsl::profiler::TraceMeLevel::kInfo);
  Py_ssize_t args_size = PyTuple_GET_SIZE(args);
  if (args_size < FAST_PATH_EXECUTE_ARG_INPUT_START) {
    PyErr_SetString(
        PyExc_ValueError,
        Printf("There must be at least %d items in the input tuple.",
               FAST_PATH_EXECUTE_ARG_INPUT_START)
            .c_str());
    return nullptr;
  }

  FastPathOpExecInfo op_exec_info;

  PyObject* py_eager_context =
      PyTuple_GET_ITEM(args, FAST_PATH_EXECUTE_ARG_CONTEXT);

  // TODO(edoper): Use interned string here
  PyObject* eager_context_handle =
      PyObject_GetAttrString(py_eager_context, "_context_handle");

  TFE_Context* ctx = reinterpret_cast<TFE_Context*>(
      PyCapsule_GetPointer(eager_context_handle, nullptr));
  op_exec_info.ctx = ctx;
  op_exec_info.args = args;

```
https://github.com/tensorflow/tensorflow/blob/805ae2c26493334538f53244229cd573ba871711/tensorflow/python/eager/pywrap_tfe_src.cc#L3102
-  >>>>> 

 ```cpp
 class ArgMinOp
    : public ArgOp<Device,
```
not much focused on this code
 https://github.com/tensorflow/tensorflow/blob/805ae2c26493334538f53244229cd573ba871711/tensorflow/core/kernels/argmax_op.cc#L127
 
- >>>>>
```cpp
struct ArgMin {
#define DECLARE_COMPUTE_SPEC(Dims)                                             \
  EIGEN_ALWAYS_INLINE static void Reduce##Dims(                                \
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,            \
      const int32 dimension, typename TTypes<Tout, Dims - 1>::Tensor output) { \
    output.device(d) = input.argmin(dimension).template cast<Tout>();          \
  }

```
https://github.com/tensorflow/tensorflow/blob/805ae2c26493334538f53244229cd573ba871711/tensorflow/core/kernels/argmax_op.h#L54


Then go into Eigen value
- > 
```cpp
static void test_argmin_tuple_reducer()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  tensor = (tensor + tensor.constant(0.5)).log();

  Tensor<Tuple<DenseIndex, float>, 4, DataLayout> index_tuples(2,3,5,7);
  index_tuples = tensor.index_tuples();

  Tensor<Tuple<DenseIndex, float>, 0, DataLayout> reduced;
  DimensionList<DenseIndex, 4> dims;
  reduced = index_tuples.reduce(
      dims, internal::ArgMinTupleReducer<Tuple<DenseIndex, float> >());

  Tensor<float, 0, DataLayout> mini = tensor.minimum();

  VERIFY_IS_EQUAL(mini(), reduced(0).second);

  array<DenseIndex, 3> reduce_dims;
  for (int d = 0; d < 3; ++d) reduce_dims[d] = d;
  Tensor<Tuple<DenseIndex, float>, 1, DataLayout> reduced_by_dims(7);
  reduced_by_dims = index_tuples.reduce(
      reduce_dims, internal::ArgMinTupleReducer<Tuple<DenseIndex, float> >());

  Tensor<float, 1, DataLayout> min_by_dims = tensor.minimum(reduce_dims);

  for (int l = 0; l < 7; ++l) {
    VERIFY_IS_EQUAL(min_by_dims(l), reduced_by_dims(l).second);
  }
}
```
https://github.com/libigl/eigen/blob/1f05f51517ec4fd91eed711e0f89e97a7c028c0e/unsupported/Eigen/CXX11/src/Tensor/TensorFunctors.h#L423

- >>
```cpp
   EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorTupleReducerOp<
      internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >,
      const array<Index, NumDimensions>, const Derived>
    argmax() const {
      array<Index, NumDimensions> in_dims;
      for (int d = 0; d < NumDimensions; ++d) in_dims[d] = d;
      return TensorTupleReducerOp<
        internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >,
        const array<Index, NumDimensions>,
        const Derived>(derived(), internal::ArgMaxTupleReducer<Tuple<Index, CoeffReturnType> >(), -1, in_dims);
    }

    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    const TensorTupleReducerOp<
      internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >,
      const array<Index, NumDimensions>, const Derived>
    argmin() const {
      array<Index, NumDimensions> in_dims;
      for (int d = 0; d < NumDimensions; ++d) in_dims[d] = d;
      return TensorTupleReducerOp<
        internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >,
        const array<Index, NumDimensions>,
        const Derived>(derived(), internal::ArgMinTupleReducer<Tuple<Index, CoeffReturnType> >(), -1, in_dims);
    }

```
https://github.com/libigl/eigen/blob/1f05f51517ec4fd91eed711e0f89e97a7c028c0e/unsupported/Eigen/CXX11/src/Tensor/TensorBase.h#L631




