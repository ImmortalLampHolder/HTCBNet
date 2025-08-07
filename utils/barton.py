import numpy as np

def barton_preprosses(data):
    data = np.array(data,dtype=np.float32)
    data = data - np.min(data)
    data = data * 100 / 400
    return data

