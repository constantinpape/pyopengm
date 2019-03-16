#include "pybind11/pybind11.h"


namespace py = pybind11;

namespace opengm_wrapper {

    void exportGM(py::module & m) {
        // TODO
        typedef GM;
        m.class<GM>(m, "GraphicalModel")
            .def(py::init<const std::size_t, const std::size_t>())
    }


}
