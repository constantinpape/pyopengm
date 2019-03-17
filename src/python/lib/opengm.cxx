#include <pybind11/pybind11.h>
#include <iostream>

// IMPORTANT: This define needs to happen the first time that pyarray is
// imported, i.e. RIGHT HERE !
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

namespace py = pybind11;


namespace pyopengm {
    void exportGM(py::module &);
    void exportFid(py::module &);
}


PYBIND11_MODULE(_opengm, module) {

    xt::import_numpy();
    module.doc() = "modern opengm python bindings";

    using namespace pyopengm;
    exportGM(module);
    exportFid(module);
}
