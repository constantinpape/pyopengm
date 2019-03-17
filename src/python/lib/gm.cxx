#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor-python/pytensor.hpp"

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
        typedef typename GM::FunctionIdentifier FunctionIdentifier;
        typedef typename GM::ValueType GmValueType;

        py::class_<GM>(m, "_GraphicalModelAdder")
            // empty constructor
            .def(py::init<>())

            // constructor from states
            .def(py::init([](const std::vector<GmIndexType> & states,
                             const std::size_t reserveFactors){

                typedef typename GM::SpaceType GmSpace;
                return std::unique_ptr<GM>(new GM(GmSpace(states.begin(), states.end()), reserveFactors));
            }), py::arg("states"),
                py::arg("reserveFactors")=1)

            // add factors
            .def("_addFactors_vector", [](GM & gm,
                                          const std::vector<FunctionIdentifier> & fids,
                                          const xt::pytensor<GmIndexType, 2> & vis,
                                          const bool finalize){
                typedef typename GM::FunctionIdentifier FidType;
                typedef typename GM::IndexType IndexType;
                const std::size_t numFid = fids.size();
                const std::size_t numVis = vis.shape()[0];
                const std::size_t factorOrder=vis.shape()[1];
                if(numFid != numVis && numFid != 1) {
                   throw std::runtime_error("len(fids) must be 1 or len(vis)");
                }
                FidType fid;
                IndexType retFactorIndex=0;
                if(numFid == 1) {
                   fid=fids[0];
                }
                {
                    py::gil_scoped_release lift_gil;
                    opengm::FastSequence<IndexType, 5> visI(factorOrder);
                    for(std::size_t i = 0;i < numVis; ++i){
                       if(numFid != 1)
                          fid=fids[i];
                       for(size_t j = 0; j < factorOrder; ++j){
                          visI[j] = vis(i, j);
                       }
                       if(finalize)
                          retFactorIndex = gm.addFactor(fid, visI.begin() ,visI.end());
                       else
                          retFactorIndex = gm.addFactorNonFinalized(fid, visI.begin(), visI.end());
                    }
                }
                return retFactorIndex;
            }, py::arg("fids"), py::arg("vis"), py::arg("finalize"))

            // add unary factors
            .def("_addUnaryFactors_vector", [](GM & gm,
                                               const std::vector<FunctionIdentifier> & fids,
                                               const xt::pytensor<GmIndexType, 1> & vis,
                                               const bool finalize){
                typedef typename GM::FunctionIdentifier FidType;
                typedef typename GM::IndexType IndexType;
                const std::size_t numFid = fids.size();
                const std::size_t numVis = vis.shape()[0];
                IndexType retFactorIndex = 0;
                if(numFid != numVis && numFid != 1)
                   throw std::runtime_error("len(fids) must be 1 or len(vis)");
                {
                    py::gil_scoped_release rgil;

                    FidType fid;
                    if(numFid==1)
                        fid = fids[0];
                    for(std::size_t i = 0; i < numVis; ++i){
                       // extract fid
                       if(numFid != 1)
                          fid = fids[i];
                       const IndexType vi=vis[i];
                       if(finalize)
                          retFactorIndex = gm.addFactor(fid,&vi,&vi+1);
                       else
                          retFactorIndex=  gm.addFactorNonFinalized(fid,&vi,&vi+1);
                    }
                }
            }, py::arg("fids"), py::arg("vis"), py::arg("finalize"))

            // add unary functions
            .def("_addUnaryFuncitions_vector", [](GM & gm,
                                                  const xt::pytensor<GmValueType, 2> & view){

                typedef typename GM::FunctionIdentifier FidType;
                typedef typename GM::ValueType ValueType;
                typedef typename GM::IndexType IndexType;
                typedef typename GM::LabelType LabelType;
                typedef size_t const * ShapeIteratorType;
                typedef opengm::FastSequence<IndexType, 1> FixedSeqType;
                //typedef typename FixedSeqType::const_iterator FixedSeqIteratorType;
                typedef opengm::SubShapeWalker<ShapeIteratorType, FixedSeqType, FixedSeqType> SubWalkerType;
                typedef opengm::ExplicitFunction<typename GM::ValueType, typename GM::IndexType, typename GM::LabelType> ExplicitFunction;

                const std::size_t numF = view.shape()[0];
                const std::size_t numLabels = view.shape()[1];
                std::vector<typename GM::FunctionIdentifier> fidVec(numF);
                {
                    py::gil_scoped_release rgil;
                    for(std::size_t f = 0; f < numF; ++f){
                        // add new function to gm (empty one and fill the ref.)
                        ExplicitFunction functionEmpty;
                        FidType fid=gm.addFunction(functionEmpty);
                        ExplicitFunction & function=gm. template getFunction<ExplicitFunction>(fid);
                        function.resize(view.shape().begin() + 1, view.shape().end());
                        fidVec[f] = fid;
                        for(std::size_t i = 0; i < numLabels; ++i){
                           // fill gm function with values
                           function(i) = view(f, i);
                        }
                    }
                }
                return fidVec;
            }, py::arg("view"))
        ;
    }
}
