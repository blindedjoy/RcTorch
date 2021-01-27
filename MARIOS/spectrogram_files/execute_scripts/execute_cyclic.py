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
#load tests ect.
from execute_scripts.column import *

RUN_LITE = False
PREDICTION_TYPE = "column"

# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)
sys.path.append(os.getcwd()) 

# get number of cpus available to job
try:
    ncpus = os.environ["SLURM_JOB_CPUS_PER_NODE"]
except KeyError:
    ncpus = multiprocessing.cpu_count()

experiment_specification = int(sys.argv[1])
size = sys.argv[3]
model_type = sys.argv[4]
activation_function = sys.argv[5]

accept_Specs = list(range(100))#[1, 2, 3, 4, 5, 100, 200, 300, 400, 500]

assert experiment_specification in accept_Specs

def liang_idx_convert(lb, ub, small = True):
    if small:
      lb = max(lb - 1, 0) #// 2
      ub = ub - 1 # // 2
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
class MyPool(multiprocessing.pool.Pool): #ThreadPool):#
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


def test(TEST, multiprocessing = False, gap = False):
    assert type(TEST) == bool
    if TEST == True:
      print("TEST")

      bounds = { 
                'n_nodes': 1000, 
                'cyclic_res_w': (-5, 1),       
                'cyclic_input_w' : (-5, 1),
                "cyclic_bias": (-5, 1),
                "leaking_rate" :   (0.001, 1)
                }

      if PREDICTION_TYPE == "block":
        if gap: 
          print("HA")
        else:
          print("on track")
          if RUN_LITE == True:
            experiment_set = [
                  {'target_frequency': 150, "split" : 0.5, 'obs_hz': 50.0, 'target_hz': 30.0}]

          else:
            experiment_set = [
                  {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 50.0, 'target_hz': 6.0},
                  {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 1000.0},
                  {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0, 'target_hz': 1000.0},
                  {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 1000.0, 'target_hz': 500.0},
                  {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 500.0,  'target_hz': 500.0},
                  {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 250.0,  'target_hz': 100.0},
                  {'target_frequency': 1000, "split" : 0.5, 'obs_hz': 50.0,  'target_hz': 25.0},

                  #{'target_frequency': 1000, "split" : 0.9, 'obs_hz': 1000.0, 'target_hz': 1000.0},
                  #{'target_frequency': 1000, "split" : 0.9, 'obs_hz': 500.0, 'target_hz': 1000.0},
                  #{'target_frequency': 1000, "split" : 0.9, 'obs_hz': 1000.0, 'target_hz': 500.0},
                  #{'target_frequency': 1000, "split" : 0.9, 'obs_hz': 500.0,  'target_hz': 500.0},
                  #{'target_frequency': 1000, "split" : 0.9, 'obs_hz': 250.0,  'target_hz': 100.0},
                  #{'target_frequency': 1000, "split" : 0.9, 'obs_hz': 100.0,  'target_hz': 100.0},
                  ]
          
          #experiment_set = [ Merge(experiment, librosa_args) for experiment in experiment_set]
        set_specific_args = {"prediction_type": "column", 
                             "size" : "publish",
                             "model_type" : "cyclic",
                             "activation_function" : "sin_sq"}

        experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]

      elif PREDICTION_TYPE == "column":

        librosa_args = {}
        
        gap_start = 250
        train_width = 50
        train_width = gap_start

        #not actually a test, we need this asap.
        zhizhuo_target1  = liang_idx_convert(gap_start, 289)  #249 -> 288 inclusive
        zhizhuo_train1   = liang_idx_convert(gap_start - train_width, gap_start - 1 ) #199 -> 248 inclusive

        subseq_len = int(np.array(zhizhuo_train1).shape[0] * 0.5)
        
        gap_start2 = 514
        zhizhuo_target2 = liang_idx_convert(gap_start2, 613) #514 -> 613 in matlab, 513 -> 612 in python
        zhizhuo_train2  = liang_idx_convert(gap_start2 - train_width, gap_start2 - 1 )



        single_column_target = liang_idx_convert(300, 375)
        single_column_train = liang_idx_convert(0, 300-1)

        print("single column target" + str(single_column_target))

        set_specific_args = {"prediction_type": "column"}
        experiment_set = [
                          {'split': 0.5, 'train_time_idx': single_column_train, 'test_time_idx': single_column_target, 'k' : None, "subseq_len" : 3},
                          {'split': 0.5, 'train_time_idx': zhizhuo_train1 , 'test_time_idx': zhizhuo_target1, 'k' : None, "subseq_len" : subseq_len},
                          {'split': 0.5, 'train_time_idx': zhizhuo_train2, 'test_time_idx':  zhizhuo_target2, 'k' : 10, "subseq_len" : subseq_len},
                          #{'split': 0.5, 'train_time_idx': single_column_train, 'test_time_idx': single_column_target, 'k' : 10, "subseq_len" : 3},
                          #{'split': 0.5, 'train_time_idx': zhizhuo_train1 , 'test_time_idx': zhizhuo_target1, 'k' : 10, "subseq_len" : subseq_len},#, "k" : 100},
                          #{'split': 0.5, 'train_time_idx': zhizhuo_train2, 'test_time_idx':  zhizhuo_target2, 'k' : 10, "subseq_len" : subseq_len},#, "k" : 30},
                          ]
        experiment_set = [ Merge(experiment, set_specific_args) for experiment in experiment_set]
    else:
      print("This is not a test")
      bounds = { 
                'n_nodes': 1000,         #fixed
                'cyclic_res_w': (-5, 1),       
                'cyclic_input_w' : (-5, 1),
                "cyclic_bias": (-5, 1),
                "leaking_rate" :   (0.001, 1)
                }

      if RUN_LITE == True:
        bounds = { 
                'n_nodes': 1000,         #fixed
                'cyclic_res_w': (-5, 1),       
                'cyclic_input_w' : (-5, 1),
                "cyclic_bias": (-5, 1),
                "leaking_rate" :   (0.001, 1)
                }
    
      obs_freqs, resp_freqs   = get_frequencies(1)
      obs_freqs2, resp_freqs2 = get_frequencies(2)
      obs_freqs3, resp_freqs3 = get_frequencies(3)
      obs_freqs4, resp_freqs4 = get_frequencies(4)
      obs_freqs5, resp_freqs5 = get_frequencies(5)
      obs_freqs6, resp_freqs6 = get_frequencies(6)
      obs_freqs7, resp_freqs7 = get_frequencies(7)

      experiment_set = [
             #{ 'split': 0.9, "obs_freqs": obs_freqs6, "target_freqs": resp_freqs6 },
             #{ 'split': 0.9, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
             #{ 'split': 0.9, "obs_freqs": obs_freqs7, "target_freqs": resp_freqs7 },
             
             #{ 'split': 0.7, "obs_freqs": obs_freqs6, "target_freqs": resp_freqs6 },
             #{ 'split': 0.7, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
             #{ 'split': 0.7, "obs_freqs": obs_freqs7, "target_freqs": resp_freqs7 },
             
             { 'split': 0.5, "obs_freqs": obs_freqs6, "target_freqs": resp_freqs6 },
             { 'split': 0.5, "obs_freqs": obs_freqs3, "target_freqs": resp_freqs3 },
             { 'split': 0.5, "obs_freqs": obs_freqs7, "target_freqs": resp_freqs7 },
             ]
    #size_ = "small"
    for experiment in experiment_set:
      experiment["bounds"] = bounds
      experiment["size"] = size
      experiment["model_type"] = model_type
      experiment["activation_function"] = activation_function

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
  print("RUNNING EXPERIMENT " + str(experiment_specification) + " YOU ARE NOT RUNNING EXP TESTS RIGHT NOW")


  TEST = True #False for low frequencies, true for column, 1000 hz

  start = timeit.default_timer()
  test(TEST = TEST)
  stop = timeit.default_timer()
  print('Time: ', stop - start) 