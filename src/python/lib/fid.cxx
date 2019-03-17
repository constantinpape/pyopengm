#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "opengm/graphicalmodel/graphicalmodel.hxx"

namespace py = pybind11;
namespace pyopengm {

    void exportFid(py::module & m){
       typedef opengm::UInt64Type FunctionIndexType;
       typedef opengm::UInt8Type FunctionTypeIndexType;
       typedef opengm::FunctionIdentification<FunctionIndexType, FunctionTypeIndexType> PyFid;

       py::class_<PyFid>(m, "FunctionIdentifier")
               .def(py::init<const FunctionIndexType, const FunctionTypeIndexType>())
               .def("getFunctionType", &PyFid::getFunctionType)
               .def("getFunctionIndex", &PyFid::getFunctionIndex)
               .def_property_readonly("functionType", &PyFid::getFunctionType)
               .def_property_readonly("functionIndex", &PyFid::getFunctionIndex)
       ;
    }
}
