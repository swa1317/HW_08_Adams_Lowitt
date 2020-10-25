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

#
# Loop through all attribute/class pairs and find the cross correlation. Print the results
#
def get_crss_corr_coef(data):
    num_cols = data.shape[1] # number of columns in the data
    for first_index in range(1,num_cols): # loop through all columns skipping the first column (ID)
        for second_index in range(1,num_cols):
            array1 = data[:,first_index] # get the array of the first attribute/class column
            array2 = data[:,second_index] # get the array of the second attribute/class column
            CrossCoef_AB = np.corrcoef(array1.ravel(), array2.ravel()) # compute cross correlation coefficient using numpy.corrcoef()
            print("Cross correlation between indexes: " + str(first_index) + " and " + str(second_index) + " = " + str(CrossCoef_AB[0,1]) )

if __name__ == '__main__':
    parameter = sys.argv[1:]
    if len(parameter) == 0: # check for an empty parameter
        print("the parameter is empty")
    else:
        data = csv_to_array(parameter[0]) # call the csv_to_array method to get a numpy 2d representation of the data
        get_crss_corr_coef(data) # call the get_crss_corr_coef method to get the cross-correlation coefficient of each class pair