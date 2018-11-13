import numpy as np
import pandas as pd

class Normalizer(object):
    w = None

    def __init__(self):
        pass

    def normalize(self, X):
        self.w = np.sqrt(X.pow(2).sum())
        return X.divide(self.w)

    def reverse_normalize(self, X):
        return X.mul(self.w)
        


