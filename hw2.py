from collections import namedtuple
from cvxpy import *
import logging
import matplotlib.pyplot as plt
import numpy as np
import time

logging.basicConfig(filename="hw2.log", level=logging.DEBUG)

#DATA GENERATION
def generate_data():
  return None, None

X, n = generate_data()

def f(b, X=X):
  return 0.5 * np.dot(b.t, np.dot(X, b))

def grad_f(b, X=X):
  return np.dot(X, b)

def main(eta, X=X, n=n, grad_f=grad_f, tolerance=1e-3, failure_cutoff=1000):
  logging.info("Running Algorithm with eta={eta}".format(eta=eta))
  path = []
  b0 = np.ones((n,1)) #start off with all ones as directed
  b = b0
  iterations = 0
  while np.linalg.norm(b) > tolerance:
    path.append(b)
    b_next = b - eta * grad_f(b)
    b = b_next
    iterations += 1
    if iterations > failure_cutoff:
      logging.info("Algorithm failed to converge with tolerance {tol} in {n} iterations".format(tol=tolerance, n=failure_cutoff))
      return None
  logging.info("Recovered optimal solution to within {tol} in {n} iterations.".format(tol=tolerance, n=iterations))
  return b_next, path

converged = [] #List[(eta, final result, path)]
diverged = []
for eta in np.linspace(0, 5, 100):
  res = main(eta)
  if res[0] is not None:
    converged.append((eta, res))
  else:
    converged.append((eta, res))
  
logging.info("Algorithm converged for eta from {min_} to (max_}".format(min_=min(converged), max_=max(converged)))
logging.info("Algorithm diverged for eta from {min_} to (max_}".format(min_=min(diverged), max_=max(diverged)))

ex_converged = converged[0] 
ex_diverged = diverged[0]

def q1b(res): #res: List[(eta, final result, path)]
  """plot the path taken to find the optimum value."""
  eta, final, path = res
  vals = [(n,f(b)) for n, b in enumerate(path)]
  plt.plot(*zip(*vals))
  plt.show()

q1b(ex_converged)
q1b(ex_diverged)
