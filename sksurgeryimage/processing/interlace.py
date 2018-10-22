import numpy as np

def interlace(left, right):

    if not isinstance(left, np.ndarray):
        raise TypeError('Left input is not a numpy array')
        return False
    
    if not isinstance(right, np.ndarray):
        raise TypeError('Right input is not a numpy array')
        return False

    if left.shape != right.shape:
        raise ValueError('Left and Right arrays do not have the same dimensions')
        return False

    new_height = left.shape[0] + right.shape[0]
    new_dims = (new_height, left.shape[1], left.shape[2])

    interlaced = np.empty( new_dims, dtype = left.dtype)

    interlaced[0::2] = left
    interlaced[1::2] = right

    return interlaced

    
def deinterlace(interlaced):

    if not isinstance(interlaced, np.ndarray):
        raise TypeError('Input is not a numpy array')
        return False

    if interlaced.shape[0] % 2:
        raise ValueError("Interlaced array has an odd number of rows")
        return False
    

    # Using // for integer division
    output_dims = (interlaced.shape[0]//2, interlaced.shape[1], interlaced.shape[2])

    left = np.empty(output_dims, dtype = interlaced.dtype)
    right = np.empty(output_dims, dtype = interlaced.dtype)

    left = interlaced[0::2]
    right = interlaced[1::2]

    return left, right