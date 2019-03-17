import numpy as np
from ._opengm import _GraphicalModelAdder, FunctionIdentifier
# TODO more specific imports
from ._inference import *


# TODO instead of a wrapper class, we should just monkey patch
class GraphicalModelAdder():

    def __init__(self, numberOfLabels, reserveNumFactorsPerVariable):
        self._gm = _GraphicalModelAdder(numberOfLabels, reserveNumFactorsPerVariable)

    def addFactors(self, fids, variableIndices, finalize=True):

        # process the function ids
        if isinstance(fids, FunctionIdentifier):
            fidVec = FidVector()
            fidVec.append(fids)
            fids=fidVec
        elif isinstance(fids, list):
            fidVec = FidVector(fids)
            fids = fidVec

        # process the variable ids
        if (isinstance(variableIndices, np.ndarray)):
            ndim = variableIndices.ndim
            if(ndim == 1):
                return self._gm._addUnaryFactors_vector(fids, np.require(variableIndices, dtype=index_type), finalize)
            elif(ndim == 2):
                return self._gm._addFactors_vector(fids, np.require(variableIndices, dtype=index_type), finalize)
        else:
            raise NotImplementedError

    # TODO figure out which of these we actually need!
    def addFunctions(self, functions):
        if isinstance(functions, np.ndarray):
            if functions.ndim == 2:
                return self._addUnaryFunctions_numpy(numpy.require(functions,dtype=value_type))
            else:
                return self._addFunctions_numpy(numpy.require(functions,dtype=value_type))
        elif isinstance(self,list):
          return self._addFunctions_list(functions)
        else:
          try:
            return self._addFunctions_vector(functions)
          except:
            try:
              return self._addFunctions_generator(functions)
            except:
              raise RuntimeError( "%s is an not a supported type for addFunctions "%(str(type(functions)),) )


def graphicalModel(numberOfLabels,
                   operator='adder',
                   reserveNumFactorsPerVariable=0):
    """ Factory function to construct a graphical model.
    Construct a gm with ``\'adder\'`` as operator::
       >>> import opengm
       >>> gm=opengm.graphicalModel([2,2,2,2,2],operator='adder')
       >>> # or just
       >>> gm=opengm.graphicalModel([2,2,2,2,2])

    Construct a gm with ``\'multiplier\'`` as operator::
       gm=opengm.graphicalModel([2,2,2,2,2],operator='multiplier')

     Args:
         numberOfLabels : number of label sequence (can be a list or  a 1d numpy.ndarray)
         operator : operator of the graphical model. Can be 'adder' or 'multiplier' (default: 'adder')
    """
    # TODO figure out the correct label_type
    label_type = 'uint32'
    if isinstance(numberOfLabels, np.ndarray):
        numL = np.require(numberOfLabels, dtype=label_type)
    else:
        numL = numberOfLabels
    if operator == 'adder' :
        return GraphicalModelAdder(numL, reserveNumFactorsPerVariable)
    elif operator=='multiplier' :
        raise NotImplementedError()
        # return GraphicalModelMultiplier(numL, reserveNumFactorsPerVariable)
    else:
        raise NameError('operator must be \'adder\' or \'multiplier\'')

gm = graphicalModel


def pottsFunctions(shape, valueEqual, valueNotEqual):
  order = len(shape)
  numL0 = np.array([int(shape[0])], dtype=label_type)
  numL1 = np.array([int(shape[1])], dtype=label_type)

  if order == 2:
    return PottsFunctionVector(numL0, numL1,
                               np.require(valueEqual, dtype=value_type),
                               np.require(valueNotEqual, dtype=value_type))
  elif order > 2:
    raise RuntimeError("not yet implemented")
  elif order < 2:
    raise RuntimeError("len(shape)>=2 is violated")
