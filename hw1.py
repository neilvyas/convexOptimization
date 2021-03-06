from __future__ import division #very important for properly instatiating sparse matrices!
from collections import namedtuple
from cvxpy import *
import datetime
import logging
import numpy as np
import numpy.ma as ma 
import scipy.sparse
import time

#log file for each run.
#dateTag = datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
#logging.basicConfig(filename='hw1_{}.log'.format(dateTag),level=logging.DEBUG)
#this is more sophisticated than necessary
logging.basicConfig(filename="hw1.log", level=logging.DEBUG)

### DATA GENERATION
np.random.seed(1)
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

#ACTUAL HOMEWORK HAPPENS HERE!
def lsq(params):
  m,n,A,b = params.m, params.n, params.A, params.b
  x = Variable(n)
  objective = Minimize(sum_entries(square(A*x - b)))
  constraints = []
  problem = Problem(objective, constraints)
  problem.solve()
  return x.value

def lasso(params, lambda_ = 0.1):
  m,n,A,b = params.m, params.n, params.A, params.b
  x = Variable(n)
  objective = Minimize(sum_entries(square(A*x - b)) + norm(lambda_ * x, 1))
  constraints = []
  problem = Problem(objective, constraints)
  problem.solve()
  x0 = x.value
  # we note some sparsity information here because it is of interest.
  logging.info("Set lambda to {lambda_}".format(lambda_=lambda_))
  logging.info("The noisy sparse target is: {b}".format(b=b[b > 1e-8]))
  logging.info("The support of B is: {x0hat}".format(x0hat=x0[x0 > 1e-8]))
  return x0

def omp(params, sparsity=5):
  m,n,A,b = params.m, params.n, params.A, params.b
  #instantiate stuff.
  i = 0
  x = np.zeros((n,1)) #start off with no explanatory components.
  Xhat = np.zeros((m,n)) #as above.
  
  while i < sparsity:
    residual = b - np.dot(A, x)
    projection = A.T.dot(residual)
    best_component = np.argmax(projection)

    #should replace this with masks, but for now I'm going to steal the hack sid has.
    Xhat[:,best_component] = A[:,best_component]
    x = np.dot(np.linalg.pinv(Xhat), b) #introduce the new components into the solution vector.
    logging.debug("In iteration {i} the components of x are {x}".format(i=i, x=x[x > 0]))
    i += 1

  #blah blah DRY fix this later.
  logging.info("Set sparsity to {sparsity}".format(sparsity=sparsity))
  logging.info("The noisy sparse target is: {b}".format(b=b[b > 1e-8]))
  logging.info("The support of B is: {xhat}".format(xhat=x[x > 1e-8]))
  return x

procs = [lsq, lasso, omp]

###RUNNING
def run_procedure(data, proc, **kwargs):
  logging.info("\n")
  train_params, test_params = train_test(data)
  t0 = time.clock()
  soln = proc(train_params, **kwargs)
  train_time = time.clock() - t0
  logging.info("Ran on data that was {m}x{n} (m x n)".format(m=data.m, n=data.n))
  logging.info("running procedure {proc} took {train_time}s".format(proc=proc.__name__, train_time=train_time))
  train_err = test_proc(TestParams(train_params.A, train_params.b), soln)
  test_err =  test_proc(test_params, soln)
  logging.info("Train error was {err}".format(err=train_err))
  logging.info("Test error was {err}".format(err=test_err))
  b = data.beta
  logging.info("The actual sparse target is: {b}".format(b=b[b > 0]))
  return None

if __name__ == "__main__":
  #data = generate_data_set(small)
  #run_procedure(data, lsq)
  #run_procedure(data, lasso)
  #run_procedure(data, omp)

  for data in generate_data(configs):
    for proc in procs:
      try: 
        run_procedure(data, proc)
      except Exception as err:
        logging.error("The optimization routine failed to terminate with exception: {err}".format(err=err))
  
