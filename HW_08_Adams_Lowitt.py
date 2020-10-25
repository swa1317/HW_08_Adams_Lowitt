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

def get_crss_corr_coef():
    Array_A = np.array([[], [], []])
    Array_B = np.array([[], [], []])
    CC_AB = np.corrcoef(Array_A.ravel(), Array_B.ravel())

if __name__ == '__main__':
    parameter = sys.argv[1:]
    if len(parameter) == 0:
        print("the parameter is empty")
    else:
        data = csv_to_array(parameter[0])
        # get_crss_corr_coef()