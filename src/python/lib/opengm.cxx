#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;


namespace pyopengm {
    void exportGM(py::module &);
}


PYBIND11_MODULE(_opengm, module) {

    module.doc() = "modern opengm python bindings";

    using namespace pyopengm;
    exportGM(module);
}
