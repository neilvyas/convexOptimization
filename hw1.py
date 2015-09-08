from collections import namedtuple
from cvxpy import *
import datetime
import logging
import numpy as np
import scipy.sparse

#log file for each run.
dateTag = datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
logging.basicConfig(filename='hw1_{}.log'.format(dateTag),level=logging.DEBUG)

### DATA GENERATION
#container classes.
DataConfig = namedtuple('DataConfig', ('m', 'n', 'd', 'sigma'))
Data = namedtuple('Data', ('m','n','beta','y','x','X_test','y_test'))

#constants.
d = 5 #sparsity
sigma = 1e-2

#configurations
small = DataConfig(50, 500, d, sigma)
large = DataConfig(500, 5000, d, sigma)
huge = DataConfig(5000, 50000, d, sigma)
configs = [small, large, huge]


#note `config` here refers to a DataConfig
def generate_data_set(config):
  #By Sidharth Kapur
  m = config.m
  n = config.n
  sigma = config.sigma

  X = np.random.randn(m,n)

  # should create a random sparse matrix with d non-zero values,
  # but that might be off-by-one because of rounding error
  beta = scipy.sparse.rand(n,1,density = d/n).toarray()

  y = np.dot(X, beta) + np.dot(sigma, np.random.randn(m,1))

  X_test = np.random.randn(100,n)
  y_test = np.dot(X_test, beta)

  return Data(m,n,X,beta,y,X_test,y_test)

#utility for when we run everything
def generate_data(configs):
  return [generate_data_set(config) for config in configs]

###IMPLEMENTING METHODS
def lsq(data):
  pass

def lasso(data):
  pass

def omp(data):
  pass


###RUNNING
if __name__ == "__main__":
  generate_data_set(small)
