
### This software is distributed as-is without any warranty.
### Please see the CC0 1.0 Universal License distributed in this repository or visit https://creativecommons.org/publicdomain/zero/1.0/.

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')

import sys

def clean_and_discretize(data, scale):
  data[data<0] = 0
  data=np.floor(data*scale)
  return(data)

SCALE = 250
d_null = np.loadtxt(sys.argv[1])
d_0 = np.loadtxt(sys.argv[2])
POPULATION = sys.argv[3]
d_null = clean_and_discretize(d_null, SCALE)
d_0 = clean_and_discretize(d_0, SCALE)

def visualize_model_fit(data, parameters):
  #plot data histogram
  sns.distplot(data, kde=False, hist_kws={"density":True}) 
  #plot model components
  xs = np.arange( 0, (max(data)+1))
  #print(xs)
  y1s = np.ones_like( xs )
  y2s = np.ones_like( xs )
  #print(y1s)
  for i, x in enumerate( xs ):
    y1s[ i ] = parameters[0]*stats.nbinom.pmf(x, 
                                          parameters[1], 
                                          parameters[2])
    y2s[ i ] = (1-parameters[0])*stats.nbinom.pmf(x, 
                                              parameters[3], 
                                              parameters[4])
    #print(y1s)
  #print(y2s)
  plt.plot(xs,y1s, label='Human to Neanderthal', color='b')
  plt.plot(xs,y2s, label='Neanderthal to Human', color='r')
  plt.plot(xs,y1s+y2s, label='Mixture Model', color='k')
  plt.legend();

# evaluate null model
def get_null_lnl(data, params):
  pt_lnlike = lambda x: np.log(stats.nbinom.pmf(x, params[0], params[1]))
  lnlike = np.sum(
      np.apply_along_axis(pt_lnlike, axis = 0, arr = data))
  return (lnlike)

# take a look
ns = np.arange(0,12)
lnls= [get_null_lnl(d_null, [n, 0.13]) for n in ns]
#plt.plot(ns, lnls)

# optimize null model
def get_null_dist(data, x0 = [7, 0.13]):
  objective = lambda parameters: -(get_null_lnl(data, parameters)) #function to minimize
  results = optimize.minimize(objective, 
                              x0, 
                              method = 'SLSQP', 
                              bounds = ((0, None), 
                                        (1e-9, 1)))
  return(results.x)

null_params = get_null_dist(d_null)
print(null_params)

# function for evaluating a single data point given a model
def get_point_lnlike(datum, r, n1, p1, null_params):
  [ n2, p2 ] = null_params
  likelihood = r*stats.nbinom.pmf(datum, n1, p1) + (1-r)*stats.nbinom.pmf(datum, n2, p2)
  return( np.log(likelihood))

# function for evaluating the log-likelihood of a population given a model
def get_population_lnlike(pop_data, r, n1, p1, null_params):
  lnlike=np.sum(np.apply_along_axis(get_point_lnlike, 
                                    0, 
                                    pop_data, 
                                    r, 
                                    n1, 
                                    p1, 
                                    null_params)) 
  return(lnlike)

# optimize mixed model
def get_mixed_dist(data, null_params, x0 = [0.25, 1, 0.1]):
  def objective(parameters):
    [ r, n1, p1 ] = parameters
    return -(get_population_lnlike(data, r, n1, p1, null_params))
  results = optimize.minimize(objective, 
                              x0, 
                              method = 'SLSQP', 
                              bounds = ((0, 1),
                                  (0, None), 
                                  (1e-9, 1)))
  return(results.x)

data = d_0
ml = get_mixed_dist(data, null_params)
print(ml)
visualize_model_fit(data, 
                    [ml[0], 
                     ml[1], 
                     ml[2], 
                     null_params[0], 
                     null_params[1]])
filename = "FIT_" + POPULATION + ".pdf"
plt.savefig(filename)
plt.clf()

# optimize mixed model for a given r parameter
def get_mixed_dist_r(data, null_params, r, x0 = [1, 0.1]):
  def objective(parameters):
    [ n1, p1 ] = parameters
    return -(get_population_lnlike(data, r, n1, p1, null_params))
  results = optimize.minimize(objective, 
                              x0, 
                              method = 'SLSQP', 
                              bounds = ((0, None), 
                                        (1e-9, 1)))
  return(results.x)

# generate confidence intervals on r
def get_CI_r(data, null_params, resolution=100):
  rs = np.linspace(0, 1, num=resolution)
  llns = np.zeros_like(rs)
  for i, r in enumerate (rs):
    [ n1, p1 ] = get_mixed_dist_r(data, null_params, r)
    llns[i] = get_population_lnlike(data, r, n1, p1, null_params)
  CI_rs=rs[llns>=np.amax(llns)-2]
  return(np.amin(CI_rs), np.amax(CI_rs))

CI_VALS = get_CI_r(d_0, null_params)
print(CI_VALS)

# Probablilty that a single datum is from alt. distribution, not null
def get_p_alt_dist(datum, r, n1, p1, null_params):
  [ n2, p2 ] = null_params
  numerator = r*stats.nbinom.pmf(datum, n1, p1)
  denominator = numerator + (1-r)*stats.nbinom.pmf(datum, n2, p2)
  return(numerator/denominator)

VALS_PREDICT = get_p_alt_dist(data, ml[0], ml[1], ml[2],null_params)
VALS_PREDICT_Lower = get_p_alt_dist(data, CI_VALS[0], ml[1], ml[2],null_params)
VALS_PREDICT_Upper = get_p_alt_dist(data, CI_VALS[1], ml[1], ml[2],null_params)

write_out1 = open(sys.argv[4], "w")
for VAL in VALS_PREDICT:
  write_out1.write(str(VAL) + "\n")

write_out2 = open(sys.argv[5], "w")
for VAL in VALS_PREDICT_Lower:
  write_out2.write(str(VAL) + "\n")

write_out3 = open(sys.argv[6], "w")
for VAL in VALS_PREDICT_Upper:
  write_out3.write(str(VAL) + "\n")




