import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_crss_corr_coef():
    Array_A = np.array([[], [], []])
    Array_B = np.array([[], [], []])
    CC_AB = np.corrcoef(Array_A.ravel(), Array_B.ravel())