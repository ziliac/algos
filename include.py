# mute warnings
import warnings
warnings.filterwarnings("ignore")

# basic packages
import os, sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.pylab as plt
import collections
import seaborn as sns
from statsmodels import tsa
from datetime import datetime
import sklearn


# frequently used functions
from matplotlib.pylab import subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# stats & machine learning tools
from sklearn import datasets
from sklearn.datasets import make_blobs
import sklearn.linear_model as lm
import statsmodels.api as sm
import statsmodels.formula.api as sfm

# seaborn style
sns.set_style('darkgrid')



# contour map of a 2-D function
def contour(F, xlim=[-1, 1], ylim=[-1, 1], resolution=100, figsize=(10, 10), **args):
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xx, yy)
    zz = F([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=figsize)
    ax.contourf(xx, yy, zz, **args)
    return 