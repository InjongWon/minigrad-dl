/*
 * minigrad/backend_ndarray/ndarray_backend_cuda.cu
 * =================================================
 * CUDA NDArray backend (pybind11 module ndarray_backend_cuda).
 *
 * Same host-side ops as CPU for bootstrapping; replace with CUDA kernels where
 * needed. Exposes via pybind11:
 *   Compact / EwiseSetitem / ScalarSetitem
 *   EwiseMul / ScalarMul / EwiseDiv / ScalarDiv / ScalarPower
 *   EwiseMaximum / ScalarMaximum / EwiseEq / ScalarEq / EwiseGe / ScalarGe
 *   EwiseLog / EwiseExp / EwiseTanh
 *   Matmul (naive) / MatmulTiled (cache-friendly tiled)
 *   ReduceMax / ReduceSum
 *
 * Based on CMU 10-714 HW3.
 */
#include <algorithm>
#include <cmath>

// If `slots` / `signals` are macros (Qt, etc.), pybind11 module init breaks with errors like
// “no matching function for call to ‘init_slots’” because the macro expands inside pybind11 headers.
#if defined(slots)
#undef slots
#endif
#if defined(signals)
#undef signals
#endif

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>

#define TILE 4

// ---------------------------------------------------------------------------
// Array class
// ---------------------------------------------------------------------------
struct Array {
  float *ptr;
  size_t size;

  Array(size_t size) : size(size) {
    ptr = new float[size](); // zero-init
  }
  ~Array() { delete[] ptr; }

  pybind11::array_t<float> numpy() {
    return pybind11::array_t<float>({size}, {sizeof(float)}, ptr,
                                    pybind11::cast(this) // keep Array alive
    );
  }

  void fill(float val) { std::fill(ptr, ptr + size, val); }

  size_t __len__() const { return size; }
};

// ---------------------------------------------------------------------------
// Helper: iterate over a strided multi-dim array
// Calls callback(flat_index, linear_count) for each element.
// ---------------------------------------------------------------------------
template <typename Callback>
void iterate_strided(const std::vector<size_t> &shape,
                     const std::vector<ptrdiff_t> &strides, size_t offset,
                     Callback cb) {
  size_t ndim = shape.size();
  size_t total = 1;
  for (auto s : shape)
    total *= s;

  std::vector<size_t> idx(ndim, 0);
  for (size_t cnt = 0; cnt < total; ++cnt) {
    // compute flat index
    ptrdiff_t flat = (ptrdiff_t)offset;
    for (size_t d = 0; d < ndim; ++d)
      flat += (ptrdiff_t)idx[d] * strides[d];
    cb((size_t)flat, cnt);

    // increment index
    for (int d = (int)ndim - 1; d >= 0; --d) {
      idx[d]++;
      if (idx[d] < shape[d])
        break;
      idx[d] = 0;
    }
  }
}

void Compact(const Array &a, Array &out, std::vector<size_t> shape,
             std::vector<ptrdiff_t> strides, size_t offset) {
  iterate_strided(shape, strides, offset,
                  [&](size_t flat, size_t cnt) { out.ptr[cnt] = a.ptr[flat]; });
}

void EwiseSetitem(const Array &a, Array &out, std::vector<size_t> shape,
                  std::vector<ptrdiff_t> strides, size_t offset) {
  iterate_strided(shape, strides, offset,
                  [&](size_t flat, size_t cnt) { out.ptr[flat] = a.ptr[cnt]; });
}

void ScalarSetitem(size_t size, float val, Array &out,
                   std::vector<size_t> shape, std::vector<ptrdiff_t> strides,
                   size_t offset) {
  iterate_strided(shape, strides, offset,
                  [&](size_t flat, size_t) { out.ptr[flat] = val; });
}

// ---------------------------------------------------------------------------
// Element-wise ops (macro to reduce boilerplate)
// ---------------------------------------------------------------------------
#define EWISE_OP(NAME, EXPR)                                                   \
  void NAME(const Array &a, const Array &b, Array &out) {                      \
    for (size_t i = 0; i < out.size; ++i) {                                    \
      float x = a.ptr[i], y = b.ptr[i];                                        \
      out.ptr[i] = EXPR;                                                       \
    }                                                                          \
  }

#define SCALAR_OP(NAME, EXPR)                                                  \
  void NAME(const Array &a, float scalar, Array &out) {                        \
    for (size_t i = 0; i < out.size; ++i) {                                    \
      float x = a.ptr[i], s = scalar;                                          \
      out.ptr[i] = EXPR;                                                       \
    }                                                                          \
  }

EWISE_OP(EwiseAdd, x + y)
SCALAR_OP(ScalarAdd, x + s)
EWISE_OP(EwiseMul, x *y)
SCALAR_OP(ScalarMul, x *s)
EWISE_OP(EwiseDiv, x / y)
SCALAR_OP(ScalarDiv, x / s)
SCALAR_OP(ScalarPower, std::pow(x, s))
EWISE_OP(EwiseMaximum, x > y ? x : y)
SCALAR_OP(ScalarMaximum, x > s ? x : s)
EWISE_OP(EwiseEq, (float)(x == y))
SCALAR_OP(ScalarEq, (float)(x == s))
EWISE_OP(EwiseGe, (float)(x >= y))
SCALAR_OP(ScalarGe, (float)(x >= s))

void EwiseLog(const Array &a, Array &out) {
  for (size_t i = 0; i < out.size; ++i)
    out.ptr[i] = std::log(a.ptr[i]);
}
void EwiseExp(const Array &a, Array &out) {
  for (size_t i = 0; i < out.size; ++i)
    out.ptr[i] = std::exp(a.ptr[i]);
}
void EwiseTanh(const Array &a, Array &out) {
  for (size_t i = 0; i < out.size; ++i)
    out.ptr[i] = std::tanh(a.ptr[i]);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------
void ReduceMax(const Array &a, Array &out, size_t reduce_size) {
  size_t n_out = out.size;
  for (size_t i = 0; i < n_out; ++i) {
    float m = a.ptr[i * reduce_size];
    for (size_t j = 1; j < reduce_size; ++j)
      m = std::max(m, a.ptr[i * reduce_size + j]);
    out.ptr[i] = m;
  }
}

void ReduceSum(const Array &a, Array &out, size_t reduce_size) {
  size_t n_out = out.size;
  for (size_t i = 0; i < n_out; ++i) {
    float s = 0.f;
    for (size_t j = 0; j < reduce_size; ++j)
      s += a.ptr[i * reduce_size + j];
    out.ptr[i] = s;
  }
}

// ---------------------------------------------------------------------------
// Matrix multiply — naive O(mnk)
// ---------------------------------------------------------------------------
void Matmul(const Array &a, const Array &b, Array &out, size_t m, size_t n,
            size_t k) {
  std::fill(out.ptr, out.ptr + m * n, 0.f);
  for (size_t i = 0; i < m; ++i)
    for (size_t l = 0; l < k; ++l)
      for (size_t j = 0; j < n; ++j)
        out.ptr[i * n + j] += a.ptr[i * k + l] * b.ptr[l * n + j];
}

// ---------------------------------------------------------------------------
// Tiled matrix multiply — cache-friendly TILE×TILE blocks
// Input layout (Python side):
//   a_tiled[M/T, K/T, T, T]   b_tiled[K/T, N/T, T, T]
//   out_tiled[M/T, N/T, T, T]
// ---------------------------------------------------------------------------
static inline void AlignedDot(const float *__restrict__ a,
                              const float *__restrict__ b,
                              float *__restrict__ out) {
  // Dot product of two TILE×TILE blocks, accumulating into out[TILE×TILE]
  for (int i = 0; i < TILE; ++i)
    for (int l = 0; l < TILE; ++l) {
      float av = a[i * TILE + l];
      for (int j = 0; j < TILE; ++j)
        out[i * TILE + j] += av * b[l * TILE + j];
    }
}

void MatmulTiled(const Array &a, const Array &b, Array &out, size_t m, size_t n,
                 size_t k) {
  size_t tm = m / TILE, tn = n / TILE, tk = k / TILE;
  std::fill(out.ptr, out.ptr + m * n, 0.f);
  for (size_t i = 0; i < tm; ++i)
    for (size_t j = 0; j < tn; ++j)
      for (size_t l = 0; l < tk; ++l)
        AlignedDot(a.ptr + (i * tk + l) * TILE * TILE,
                   b.ptr + (l * tn + j) * TILE * TILE,
                   out.ptr + (i * tn + j) * TILE * TILE);
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(ndarray_backend_cuda, m) {
  pybind11::class_<Array>(m, "Array")
      .def(pybind11::init<size_t>())
      .def("numpy", &Array::numpy)
      .def("fill", &Array::fill)
      .def("__len__", &Array::__len__);

  m.def("Compact", &Compact);
  m.def("EwiseSetitem", &EwiseSetitem);
  m.def("ScalarSetitem", &ScalarSetitem);

  m.def("EwiseAdd", &EwiseAdd);
  m.def("ScalarAdd", &ScalarAdd);
  m.def("EwiseMul", &EwiseMul);
  m.def("ScalarMul", &ScalarMul);
  m.def("EwiseDiv", &EwiseDiv);
  m.def("ScalarDiv", &ScalarDiv);
  m.def("ScalarPower", &ScalarPower);
  m.def("EwiseMaximum", &EwiseMaximum);
  m.def("ScalarMaximum", &ScalarMaximum);
  m.def("EwiseEq", &EwiseEq);
  m.def("ScalarEq", &ScalarEq);
  m.def("EwiseGe", &EwiseGe);
  m.def("ScalarGe", &ScalarGe);
  m.def("EwiseLog", &EwiseLog);
  m.def("EwiseExp", &EwiseExp);
  m.def("EwiseTanh", &EwiseTanh);

  m.def("ReduceMax", &ReduceMax);
  m.def("ReduceSum", &ReduceSum);
  m.def("Matmul", &Matmul);
  m.def("MatmulTiled", &MatmulTiled);
}