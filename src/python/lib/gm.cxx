#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "opengm/graphicalmodel/graphicalmodel.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/pottsg.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "opengm/functions/learnable/lpotts.hxx"
#include "opengm/functions/learnable/lunary.hxx"


namespace py = pybind11;

namespace pyopengm {

    typedef double GmValueType;
    typedef opengm::UInt64Type GmIndexType;

    template<class V,class I,class O,class F>
    struct GmGen{
       typedef opengm::DiscreteSpace<I,I> SpaceType;
       typedef opengm::GraphicalModel<V,O,F,SpaceType> type;
    };


    template<class V,class I>
    struct FTLGen{

        typedef V ValueType;
        typedef I IndexType;
        typedef I LabelType;
        typedef opengm::ExplicitFunction                      <ValueType,IndexType,LabelType> PyExplicitFunction;
        typedef opengm::PottsFunction                         <ValueType,IndexType,LabelType> PyPottsFunction;
        typedef opengm::PottsNFunction                        <ValueType,IndexType,LabelType> PyPottsNFunction;
        typedef opengm::PottsGFunction                        <ValueType,IndexType,LabelType> PyPottsGFunction;
        typedef opengm::TruncatedAbsoluteDifferenceFunction   <ValueType,IndexType,LabelType> PyTruncatedAbsoluteDifferenceFunction;
        typedef opengm::TruncatedSquaredDifferenceFunction    <ValueType,IndexType,LabelType> PyTruncatedSquaredDifferenceFunction;
        typedef opengm::SparseFunction                        <ValueType,IndexType,LabelType> PySparseFunction;
        typedef opengm::functions::learnable::LPotts          <ValueType,IndexType,LabelType> PyLPottsFunction;
        typedef opengm::functions::learnable::LUnary          <ValueType,IndexType,LabelType> PyLUnaryFunction;

        typedef typename opengm::meta::TypeListGenerator<
            PyExplicitFunction,
            PyPottsFunction,
            PyPottsNFunction,
            PyPottsGFunction,
            PyTruncatedAbsoluteDifferenceFunction,
            PyTruncatedSquaredDifferenceFunction,
            PySparseFunction,
            PyLPottsFunction,
            PyLUnaryFunction
        >::type type;
    };


     typedef GmGen<
       GmValueType,
       GmIndexType,
       opengm::Adder,
       FTLGen<GmValueType, GmIndexType>::type
    >::type   GmAdder;


    void exportGM(py::module & m) {
        typedef GmAdder GM;
        py::class_<GM>(m, "GraphicalModel")
            // empty constructor
            .def(py::init<>())

            // constructor from states
            .def(py::init([](const std::vector<GmIndexType> & states,
                             const std::size_t reserveFactors){

                typedef typename GM::SpaceType GMSpace;
                return std::unique_ptr<GM>(new GM(GMSpace(states.begin(), states.end()), reserveFactors));
            }), py::arg("states"),
                py::arg("reserveFactors")=1)
        ;
    }


}
