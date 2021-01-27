import sys
from PyFiles.experiment import *
from PyFiles.analysis import *

TEACHER_FORCING = False

colorz = {
  "header" : '\033[95m',
  "blue" : '\033[94m',
  'cyan' : '\033[96m',
  'green' : '\033[92m',
  'warning' : '\033[93m',
  'fail' : '\033[91m',
  'endc' : '\033[0m',
   'bold' :'\033[1m',
   "underline" : '\033[4m'
}
  
def printc(string_, color_) :
  print(colorz[color_] + string_ + colorz["endc"] )


def get_frequencies(trial = 1):
  """
  get frequency lists
  """
  if trial =="run_fast_publish":
      lb_targ, ub_targ, obs_hz  = 340, 350, 10
  elif trial == 1:
      lb_targ, ub_targ, obs_hz  = 210, 560, int(320 / 2)   
  elif trial == 2:
      lb_targ, ub_targ, obs_hz  = 340, 640, 280
  elif trial == 3:
      lb_targ, ub_targ, obs_hz  = 340, 350, 20#40
  elif trial == 4:
      lb_targ, ub_targ, obs_hz  = 60, 350, 40
  elif trial == 5:
      lb_targ, ub_targ, obs_hz  = 50, 200, 40
  if trial == 6:
      lb_targ, ub_targ, obs_hz  = 130, 530, 130
  if trial == 7:
      lb_targ, ub_targ, obs_hz  = 500, 900, 250
  obs_list =  list( range( lb_targ - obs_hz, lb_targ))
  obs_list += list( range( ub_targ, ub_targ + obs_hz))
  resp_list = list( range( lb_targ, ub_targ))
  return obs_list, resp_list
  
def printc(string_, color_) :
  print(colorz[color_] + string_ + colorz["endc"] )

colorz = {
  "header" : '\033[95m',
  "blue" : '\033[94m',
  'cyan' : '\033[96m',
  'green' : '\033[92m',
  'warning' : '\033[93m',
  'fail' : '\033[91m',
  'endc' : '\033[0m',
   'bold' :'\033[1m',
   "underline" : '\033[4m'
}
def run_experiment(inputs, n_cores = int(sys.argv[2]), interpolation_method = "griddata-linear"):
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


  #notes on modularizing this: column.py doesn't make sense in terms of its name. Ideally you will have block.py and column.py,
  but at the end of the day the experiment specifications should be easier to execute. Namely, they should be at the top of this file.
  }"""
  model_type = inputs["model_type"]
  size = inputs["size"]


  prediction_type = inputs["prediction_type"] 
  cv_samples = 1
  batch_size = max(n_cores // cv_samples , 1) #
  random_seed = 126

  default_presets = {
     "cv_samples" : cv_samples,
     "batch_size" : batch_size,
     "random_seed" : random_seed}

  if size == "small":
    default_presets = {
      **default_presets,
      "n_res" : 1,#2,
      "eps" : 1e-8,
      'subsequence_length' : 10,
      "initial_samples" : 1000,
      }
  elif size == "medium":
    default_presets = { #cv_samples * n_res * batch size --> n_cores. what about njobs?
      **default_presets,
      "n_res" : 1,
      "eps" : 1e-4,
      'subsequence_length' : 100,
      "initial_samples" : 100,
      "max_iterations" : 5000}
  elif size == "publish":
    default_presets = {
      **default_presets,
      "max_iterations" : 3000,
      "eps" : 1e-4,
      'subsequence_length' : 700,
      "n_res": 1,
      "initial_samples" : 300}
  if "k" in inputs:
    k = inputs["k"]
  else:
    k = None
  #default arguments
  #print("Prediction Type: " + inputs["prediction_type"])

  ####if you imported the data via librosa this will work
  if "librosa" in inputs:
    default_presets = {
      "cv_samples" : 4,
      "max_iterations" : 3000,
      "eps" : 1e-8,
      'subsequence_length' : 250,
      "initial_samples" : 1000}
    librosa_args = { "spectrogram_path": inputs["spectrogram_path"],
                     "librosa": inputs["librosa"],
                     "spectrogram_type": inputs["spectrogram_type"]}
  else:
    librosa_args = {}

  EchoArgs = { "size"    : size,  "verbose" : False}

  obs_inputs = {"split" : inputs["split"], "aspect": 0.9, "plot_split": False}

  if inputs["prediction_type"] == "column":
    train_time_idx, test_time_idx = inputs["train_time_idx"], inputs["test_time_idx"]

    experiment_inputs = { "size" : inputs["size"],
                          "target_frequency" : None,
                          "verbose" : False,
                          "prediction_type" : inputs["prediction_type"],
                          "train_time_idx" : train_time_idx,
                          "test_time_idx" : test_time_idx,
                          "k" : k,
                          "model" : model_type,
                          **librosa_args}

    print("experiment_inputs: " + str(experiment_inputs))
    experiment = EchoStateExperiment(**experiment_inputs)
    
    obs_inputs = Merge(obs_inputs, {"method" : "exact"})

    print("obs_inputs: " + str(obs_inputs))
    experiment.get_observers(**obs_inputs, get_observers_input = obs_inputs)

  elif inputs["prediction_type"] == "block" or inputs["prediction_type"] == "freqs":
    if "obs_freqs" in inputs:
      AddEchoArgs = { "obs_freqs" : inputs["obs_freqs"],
                      "target_freqs" : inputs["target_freqs"],
                      "prediction_type" : inputs["prediction_type"],
                      "model" : model_type
                    }
      EchoArgs = Merge(EchoArgs, AddEchoArgs)
    else:

      AddEchoArgs = { "target_frequency" : inputs["target_frequency"],
                      "obs_hz" : inputs["obs_hz"],
                      "target_hz" : inputs["target_hz"],
                      "model" : model_type
                    }
      EchoArgs = Merge( Merge(EchoArgs, AddEchoArgs), librosa_args)
    print(EchoArgs)
    experiment = EchoStateExperiment( **EchoArgs, EchoStateExperiment_inputs = EchoArgs)
    ### NOW GET OBSERVERS
    method = "exact" if "obs_freqs" in inputs else "freq"
    
    obs_inputs = {**obs_inputs, "method": method}
    experiment.get_observers(**obs_inputs, get_observers_input = obs_inputs)

  

  if inputs["prediction_type"] == "column":

    default_presets['esn_feedback'] = True
    if "subseq_len" in inputs:
      default_presets['subsequence_length'] = inputs["subseq_len"]
    else:
      default_presets['subsequence_length'] = 75

  if inputs["feedback"] == False:
    default_presets['esn_feedback'] = False

  printc("NCORES " + str(n_cores), 'blue')

  njobs = default_presets["batch_size"] #default_presets["batch_size"] #int(np.floor(n_cores/(default_presets["n_res"] * default_presets["batch_size"]*default_presets["cv_samples"])) * 0.9) 
  

  #njobs = max(njobs, 1)
  #assert njobs >= 1
  #njobs = 1
  #printc("njobs" + str(njobs), 'warning')
  est_cores = default_presets["batch_size"]  * default_presets["n_res"] * default_presets["batch_size"]  * default_presets["cv_samples"]
  #print("estimated core use " + str(est_cores), 'green')
  cv_args = {
      'bounds' : inputs["bounds"],
      'scoring_method' : 'tanh',
      "n_jobs" : njobs,
      "verbose" : True,
      "plot" : False, 
      **default_presets
  }
  #consider making n_jobs be a calculation based on the other shit.

  assert model_type in ["random", "cyclic", "delay_line"]

  def go_(input_weight_type, model_type = model_type, cv_args = cv_args):
    experiment.RC_CV(cv_args = cv_args, model = model_type, input_weight_type = input_weight_type)


  if model_type in ["delay_line", "cyclic"]:
    cv_args = {**cv_args, "activation_function" : "sin_sq"}

  #Consider combining cyclic and delay line
  if prediction_type == "column":
    printc("running column experiment", 'green')
    go_("uniform")
  elif model_type in ["delay_line", "cyclic"]:
    printc("running cyclic experiments", 'fail')
    go_("exponential")
    go_("uniform")
    
  else:
    printc("running random experiments", 'fail')

    go_("exponential")
    go_("uniform")
    

#https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
  

