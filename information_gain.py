# File for the information gain feature selection algorithm
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# The function which will be called
def get_features(raw_data, raw_ids):

    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
    df = pd.DataFrame(raw_data)
    target = df["person"] = raw_ids

    # cv = CountVectorizer(max_df=0.95, min_df=2,
    #                     max_features=10000, stop_words='english')
    # X_vec = cv.fit_transform(X)
    #
    # res = list(zip(cv.get_feature_names(),
    #                mutual_info_classif(X_vec, Y, discrete_features=True)
    #                ))
    return mutual_info_classif(df, target, discrete_features=True)
