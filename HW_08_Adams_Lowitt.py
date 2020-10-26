import numpy as np
import pandas as pd
import sys
import math
# import matplotlib.pyplot as plt

# data structure to hold the info for each cluster
class Cluster:
    # initializes a cluster with a list of shoppers, a size and a center point
    def __init__(self, shopper):
        self.shoppers = []
        self.shoppers.append(shopper)
        self.num_shoppers = 1
        self.center_point = shopper

    def get_cluster_size(self):
        return self.num_shoppers

    # calculates the new center of the cluster after a merge
    def recalculate_center(self):
        new_center = []
        for grocery_item in range(1, 20):
            item_total = 0
            for cluster_member in range(self.num_shoppers):
                item_total += self.shoppers[cluster_member][grocery_item]
            item_avg = item_total/self.num_shoppers
            new_center.append(item_avg)
        self.center_point = new_center

    # merges two clusters by combining their shopper lists and recalculating the new center
    def merge_clusters(self, other_cluster):
        # print("merging " + str(self.num_shoppers) + " with " + str(other_cluster.num_shoppers))
        self.shoppers = self.shoppers + other_cluster.shoppers
        self.num_shoppers = self.num_shoppers + other_cluster.num_shoppers
        self.recalculate_center()


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
    max_coef = 0 # place holder for max coefficient
    for first_index in range(1,num_cols): # loop through all columns skipping the first column (ID)
        attribute_average_correlation = 0 # place holder to keep track of the average correlation for each attribute
        for second_index in range(1,num_cols):
            if first_index != second_index: # Don't compute coefficient of an attribute against itself
                array1 = data[:,first_index] # get the array of the first attribute/class column
                array2 = data[:,second_index] # get the array of the second attribute/class column
                CrossCoef_AB = np.corrcoef(array1.ravel(), array2.ravel()) # compute cross correlation coefficient using numpy.corrcoef()
                if abs(CrossCoef_AB[0,1]) > max_coef: max_coef = abs(CrossCoef_AB[0,1]) # set mac coefficient if new max is found
                attribute_average_correlation += abs(CrossCoef_AB[0,1]) # add each coefficient for computing the average later in the code
                print("Cross correlation between indexes: " + str(first_index) + " and " + str(second_index) + " = " + str(CrossCoef_AB[0,1]))
        attribute_average_correlation /= 19 # compute the average coefficient.
        print("Average attribute cross-correlation coefficient for index " + str(first_index) + " is: " + str(attribute_average_correlation))
    print("Max Cross-correlation Coefficient: " + str(max_coef))

# calculates the euclidean distance between two cluster centers
# distance = ( sigma(x-y)^2 ) ^1/2  for each item in the cluster's center
def compute_distance(c1, c2):
    total = 0
    for item in range(1, 19):
        total = total + math.pow(c1.center_point[item] - c2.center_point[item], 2)
    euclidean_distance = math.sqrt(total)
    return euclidean_distance

# Agglomerate the data
def agglomerate(data):
    # create a cluster for each shopper
    shopping_clusters = []
    for customer in data:
        sample_customer = []
        for datapoint in range(len(customer)):
            sample_customer.append(customer[datapoint])
        new_cluster = Cluster(sample_customer)
        # print(new_cluster.shoppers)
        shopping_clusters.append(new_cluster)

    last_18_clusters = []
    while len(shopping_clusters) > 1:
        best_c1 = 0
        best_c2 = 1
        best_distance = compute_distance(shopping_clusters[0], shopping_clusters[1])
        # loop through every pair of clsters to find which pair has the smallest euclidean distance
        for cluster1_idx in range(len(shopping_clusters)):
            for cluster2_idx in range(cluster1_idx, len(shopping_clusters)):
                if cluster1_idx != cluster2_idx:
                    euclidean_distance = compute_distance(shopping_clusters[cluster1_idx], shopping_clusters[cluster2_idx])
                    # if a new shortest distance is found, update new best
                    if euclidean_distance <= best_distance:
                        best_distance = euclidean_distance
                        best_c1 = cluster1_idx
                        best_c2 = cluster2_idx
        print(str(best_c1) + " + " + str(best_c2))
        print(str(shopping_clusters[best_c1].num_shoppers) + " + " + str(shopping_clusters[best_c2].num_shoppers))
        # decide which cluster to be merged is smaller
        # merges the clusters
        # removes the smaller cluster from the list
        if shopping_clusters[best_c1].num_shoppers >= shopping_clusters[best_c2].num_shoppers:
            shopping_clusters[best_c1].merge_clusters(shopping_clusters[best_c2])
            smaller_cluster_size = shopping_clusters[best_c2].num_shoppers
            shopping_clusters.pop(best_c2)
        else:
            shopping_clusters[best_c2].merge_clusters(shopping_clusters[best_c1])
            smaller_cluster_size = shopping_clusters[best_c1].num_shoppers
            shopping_clusters.pop(best_c1)

        # print(str(best_distance))
        # if 18 clusters left, start keeping track of sizes
        if len(shopping_clusters) <= 18:
            last_18_clusters.append(smaller_cluster_size)
            for finalist_cluster in range(18):
                print("size: " + str(shopping_clusters[finalist_cluster].num_shoppers))
                print("center: " + str(shopping_clusters[finalist_cluster].center_point))
    # print("agglomerated")
    # print("last 18 smallest clusters:")
    print(last_18_clusters)


if __name__ == '__main__':
    parameter = sys.argv[1:]
    if len(parameter) == 0: # check for an empty parameter
        print("the parameter is empty")
    else:
        data = csv_to_array(parameter[0]) # call the csv_to_array method to get a numpy 2d representation of the data
        get_crss_corr_coef(data) # call the get_crss_corr_coef method to get the cross-correlation coefficient of each class pair
        agglomerate(data) # call the agglomerate method to agglomerate the data
