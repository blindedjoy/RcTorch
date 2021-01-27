#!/bin/bash/env python3

from itertools import combinations
import multiprocessing
from multiprocessing import set_start_method
import os
from PyFiles.experiment import *
from PyFiles.helpers import *
from PyFiles.imports import *
from random import randint
from reservoir import *
import sys
import time
import timeit
import multiprocessing.pool 

PREDICTION_TYPE = "block"

TEACHER_FORCING = False

# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)
sys.path.append(os.getcwd()) 

# get number of cpus available to job
try:
    ncpus = os.environ["SLURM_JOB_CPUS_PER_NODE"]
except KeyError:
    ncpus = multiprocessing.cpu_count()

experiment_specification = int(sys.argv[1])
accept_Specs = list(range(100))#[1, 2, 3, 4, 5, 100, 200, 300, 400, 500]

assert experiment_specification in accept_Specs

def liang_idx_convert(lb, ub, k = None, small = True):
    if small:
      lb = lb #// 2
      ub = ub # // 2
    idx_list = list(range(lb, ub + 1))
    return idx_list

class NoDaemonProcess(multiprocessing.Process):
      @property
      def daemon(self):
          return False

      @daemon.setter
      def daemon(self, value):
          pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
#class MyPool(multiprocessing.pool.Pool): #ThreadPool):#
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)

def run_experiment(inputs, n_cores = int(sys.argv[2]), cv_samples = 5, size = "medium",
                   interpolation_method = "griddata-linear"):
      """
      4*4 = 16 + 

      The final form of the input dict is:

        inputs = {"target_frequency_" : ...
                  "obs_hz" : ...
                  "target_hz" : ...
                  "split" : ...
                  ""
                  }

      Reinier's example:
      {
        'leaking_rate' : (0, 1), 
        'spectral_radius': (0, 1.25),
        'regularization': (-12, 1),
        'connectivity': (-3, 0),
        'n_nodes':  (100, 1000)
      }

      """
      #default arguments
      print("Prediction Type: " + inputs["prediction_type"])

      ####if you imported the data via librosa this will work
      if "librosa" in inputs:
        default_presets = { "cv_samples" : 4, "max_iterations" : 3000, "eps" : 1e-8, 
                            'subsequence_length' : 250, "initial_samples" : 1000}

        librosa_args = { "spectrogram_path": inputs["spectrogram_path"],
                         "librosa": inputs["librosa"],
                         "spectrogram_type": inputs["spectrogram_type"]
                        }
      else:
        librosa_args = {}

      EchoArgs = { "size"    : size, "verbose" : False}

      obs_inputs = {"split" : inputs["split"], "aspect": 0.9, "plot_split": False}

      if "k" in inputs:
        obs_inputs["k"] = inputs["k"]

      if PREDICTION_TYPE == "column":
        train_time_idx, test_time_idx = inputs["train_time_idx"], inputs["test_time_idx"]

        experiment_inputs = { "size" : size,
                              "target_frequency" : None,
                              "verbose" : False,
                              "prediction_type" : inputs["prediction_type"],
                              "train_time_idx" : train_time_idx,
                              "test_time_idx" : test_time_idx,
                              **librosa_args
                            }
        obs_inputs = Merge(obs_inputs, {"method" : "exact"})

        experiment = EchoStateExperiment(**experiment_inputs)
        experiment.get_observers(**obs_inputs)

        print("obs_inputs: " + str(obs_inputs))
        print("experiment_inputs: " + str(experiment_inputs))
      
      elif PREDICTION_TYPE == "block":
        if "obs_freqs" in inputs:
          AddEchoArgs = { "obs_freqs" : inputs["obs_freqs"],
                          "target_freqs" : inputs["target_freqs"],
                          "prediction_type" : PREDICTION_TYPE
                        }
          EchoArgs = Merge(EchoArgs, AddEchoArgs)
        else:
          AddEchoArgs = { "target_frequency" : inputs["target_frequency"],
                          "obs_hz" : inputs["obs_hz"],
                          "target_hz" : inputs["target_hz"]
                        }
          EchoArgs = Merge( Merge(EchoArgs, AddEchoArgs), librosa_args)
        method = "exact" if "obs_freqs" in inputs else "freq"

        print(EchoArgs)

        experiment = EchoStateExperiment( **EchoArgs)
        experiment.get_observers(method = method, **obs_inputs)
      
      if size == "small":
        default_presets = { "cv_samples" : 6, "max_iterations" : 1000, "eps" : 1e-5,
                            'subsequence_length' : 180, "initial_samples" : 100}
      elif size == "medium":
        default_presets = { "cv_samples" : 5, "max_iterations" : 4000, "eps" : 1e-5,
                            'subsequence_length' : 250, "initial_samples" : 100}
      elif size == "publish":
        default_presets = { "cv_samples" : 5, "max_iterations" : 2000, "eps" : 1e-4,
                            'subsequence_length' : 500, "initial_samples" : 200}

      if PREDICTION_TYPE == "column":
        if "subseq_len" in inputs:
          default_presets['subsequence_length'] = inputs["subseq_len"]
        else:
          default_presets['subsequence_length'] = 75

      cv_args = { 'bounds' : inputs["bounds"], 'scoring_method' : 'tanh', "n_jobs" : n_cores,
                  "verbose" : True, "plot" : False,  **default_presets}

      if TEACHER_FORCING:
        cv_args = Merge(cv_args, {"esn_feedback" : True})

      models = ["exponential", "uniform"] if PREDICTION_TYPE == "block" else ["uniform"] #
      for model_ in models:
        print("Train shape: " + str(experiment.Train.shape))
        print("Test shape: " +  str(experiment.Test.shape))
        experiment.RC_CV(cv_args = cv_args, model = model_)

def get_frequencies(trial = 1):
  """
  get frequency lists
  """
  if trial == 1:
      lb_targ, ub_targ, obs_hz  = 210, 560, int(320 / 2)
  elif trial == 2:
      lb_targ, ub_targ, obs_hz  = 340, 640, 280
  elif trial == 3:
      lb_targ, ub_targ, obs_hz  = 340, 350, 40
  obs_list =  list( range( lb_targ - obs_hz, lb_targ, 10))
  obs_list += list( range( ub_targ, ub_targ + obs_hz, 10))
  resp_list = list( range( lb_targ, ub_targ, 10))
  return obs_list, resp_list

def test(TEST, multiprocessing = False, gap = False):
    assert type(TEST) == bool
    print("TEST")
    if PREDICTION_TYPE == "block":
      bounds = { #noise hyper-parameter.
         #all are log scale except  spectral radius, leaking rate and n_nodes
        'noise' :          (-2, -4),
        'llambda' :        (-3, -1), 
        'connectivity':    (-3, 0),       # 0.5888436553555889, 
        'n_nodes':         1000,          #(100, 1500),
        'spectral_radius': 0.99,
        'regularization':  (-3, 3),#(-12, 1),
        "leaking_rate" :   0.99 # we want some memory. 0 would mean no memory.
      }
      experiment_set = [ {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 20, 'target_hz': 10}]
      set_specific_args = {"prediction_type": "block"}
      experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]

    for experiment in experiment_set:
      experiment["bounds"] = bounds

    try:
      set_start_method('forkserver')
    except RuntimeError:
      pass
    
    n_experiments = len(experiment_set)
    exper_ = [experiment_set[experiment_specification]]

    #print("Creating " + str(n_experiments) + " (non-daemon) workers and jobs in main process.")
    if n_experiments > 1:
      pool = MyPool(n_experiments)
      pool.map(run_experiment, exper_)
      pool.close()
      pool.join()
    else:
      run_experiment(exper_[0])


if __name__ == '__main__':
  print("Total cpus available: " + str(ncpus))
  print("RUNNING EXPERIMENT " + str(experiment_specification))
  TEST = True

  start = timeit.default_timer()
  test(TEST = TEST)
  stop = timeit.default_timer()
  print('Time: ', stop - start) 

