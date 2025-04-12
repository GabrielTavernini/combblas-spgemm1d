#include <cstdint>
#include <pybind11/pybind11.h>

#include "CombBLAS/SpDCCols.h"

namespace py = pybind11;

typedef int64_t IT;
typedef int64_t IT;
PYBIND11_MODULE(pycombblas, m) {
    py::class_<combblas::SpDCCols<int64_t, double>>(m,"SpDCCols_ld")
    .def(py::init<>());
}