from collections import namedtuple
from cvxpy import *
import datetime
import logging
import numpy as np
import scipy.sparse
import time

#log file for each run.
dateTag = datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
logging.basicConfig(filename='hw1_{}.log'.format(dateTag),level=logging.DEBUG)

### DATA GENERATION
#container classes.
DataConfig = namedtuple('DataConfig', ('m', 'n', 'd', 'sigma'))
Data = namedtuple('Data', ('m','n','X','beta','y','X_test','y_test'))

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
  #(mainly) By Sidharth Kapur
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
TrainParams = namedtuple('TrainingParameters', ('m', 'n', 'A', 'b'))
TestParams = namedtuple('TestingParameters', ('X', 'y'))
def train_test(data):
  m,n = data.m, data.n
  train_params = TrainParams(m, n, data.X, data.y)
  test_params = TestParams(data.X_test, data.y_test)
  return train_params, test_params

def test_proc(params, soln, fn = np.linalg.norm, **kwargs):
  X_test, y_test = params.X, params.y 
  return fn(X_test.dot(soln) - y_test)

def lsq(params):
  m,n,A,b = params.m, params.n, params.A, params.b
  x = Variable(n)
  objective = Minimize(sum_entries(square(A*x - b)))
  constraints = []
  problem = Problem(objective, constraints)
  problem.solve()
  return x.value

def lasso(data, lambda_ = 0.1):
  pass

def omp(data, sparsity=5):
  pass


###RUNNING
def run_procedure(data, proc, **kwargs):
  train_params, test_params = train_test(data)
  t0 = time.clock()
  soln = proc(train_params, **kwargs)
  train_time = time.clock() - t0
  logging.info("running procedure {proc} took {train_time}s".format(proc=proc.__name__, train_time=train_time))
  return test_proc(test_params, soln)

if __name__ == "__main__":
  data = generate_data_set(small)
  #lsq_soln = lsq(data.m, data.n, data.X, data.y)
  #print np.linalg.norm(data.X_test.dot(lsq_soln) - data.y_test)
  print run_procedure(data, lsq)
