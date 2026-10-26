
from . import model

#from . import counter_mnist
#from . import counter_random_mnist
from . import puzzle_mnist
#from . import puzzle_wrong
from . import puzzle_labeled_objects
from . import split_image

def shuffle_objects(objects):
    import numpy as np
    import numpy.random
    tmp = np.copy(objects)
    tmp = np.transpose(tmp,(1,0,2))
    np.random.shuffle(tmp)
    tmp = np.transpose(tmp,(1,0,2))
    return tmp
