import numpy as np
# TODO more specific imports
from ._opengm import *
from ._inference import *


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
        numL = numpy.require(numberOfLabels, dtype=label_type)
    else:
        numL = numberOfLabels
    if operator == 'adder' :
        return adder.GraphicalModel(numL, reserveNumFactorsPerVariable)
    elif operator=='multiplier' :
        raise NotImplementedError()
        return multiplier.GraphicalModel(numL, reserveNumFactorsPerVariable)
    else:
        raise NameError('operator must be \'adder\' or \'multiplier\'')

gm = graphicalModel
