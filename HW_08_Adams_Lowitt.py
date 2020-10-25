import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

def csv_to_array(csv_filename):
    """
    csv_filename: the name of the csv file
    return: the numpy 2d array of csv file
    """
    csv_pandas = pd.read_csv(csv_filename)
    data = csv_pandas.to_numpy()
    return data

def get_crss_corr_coef(data):
    num_cols = data.shape[1]
    for first_index in range(1,num_cols):
        for second_index in range(1,num_cols):
            array1 = data[:,first_index]
            array2 = data[:,second_index]
            CrossCoef_AB = np.corrcoef(array1.ravel(), array2.ravel())
            print("Cross correlation between indexes: " + str(first_index) + " and " + str(second_index) + " = " + str(CrossCoef_AB) )

if __name__ == '__main__':
    parameter = sys.argv[1:]
    if len(parameter) == 0:
        print("the parameter is empty")
    else:
        data = csv_to_array(parameter[0])
        get_crss_corr_coef(data)