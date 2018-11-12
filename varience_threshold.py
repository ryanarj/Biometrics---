# File for the varience threshold feature selection algorithm
import numpy as np
from sklearn.feature_selection import VarianceThreshold


def varience_threshold(data):
    varience = np.var(data)
    m = data - np.mean(varience)
    vt = VarianceThreshold(threshold=m)
    fit_data = vt.fit_transform(data)
    return fit_data


