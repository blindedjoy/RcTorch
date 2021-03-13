#Herein we shall create a new file, similar to esn.py 
#where we transform the notebook into an object oriented approach worthy of Reinier's library.

from reservoir import *
from PyFiles.imports import *
from collections import defaultdict

def Merge(dict1, dict2): 
	res = {**dict1, **dict2} 
	return res 

def nrmse(pred_, truth, columnwise = False):
	"""Calculate NRMSE (R) from numpy arrays
	
	Args:
		pred_ : the prediction numpy array
		truth : the ground truth numpy array
		columnwise: bool if set to true takes row-wise numpy array 
		(assumes reader thinks of time as running left to right while the code actually runs vertically.)
	"""
	if columnwise:
		rmse_ = np.sum((truth - pred_) ** 2, axis = 1).reshape(-1, ) #takes column-wise mean.
		denom_ = np.sum(truth ** 2) * (1/len(rmse_))
	else:
		rmse_ = np.sum((truth - pred_) ** 2)
		denom_ = np.sum(truth ** 2)
	
	nrmse_ = np.sqrt(rmse_ / denom_)
	return(nrmse_)

# From  Item 22 in Efective Python: "variable positional arguments"
def build_string(message, *values,  sep = ""):
	""" build a string with arbitrary arguments.

	Positional Args:
		message: original string to build on.
		values: individual arguments to be added to the string.
	Keyword only Args:
		sep: how to divide the parts of the string if you want. (ie. ',' or "/")
	"""
	if not values:
		return message
	else:
		return message.join(str(x)  for x in values)



def idx2Freq(freq):
	""" Takes an frequency and returns the closest index to that frequency.

	Args:
		freq: the desired Frequency to be found in the index of the experiment object.
	# TODO: decide whether or not to build this into the experiment class.
	"""
	idx = min(range(len(f)), key=lambda i: abs( f[i] - freq))
	return(idx)

def ifdel(dictt, key):
    """ If a key is in a dictionary delete it. Return [modified] dictionary.
    """
    try:
        del dictt[key]
        return(dictt)
    except:
        return(dictt)

def pp(variable, label): #TODO replace with wrappers.
	"""
	custom print function
	"""
	print(label +": " + str(variable))

def Shape(lst):
	""" Prints out the shape of an np array object
	Arguments:
		lst: (np_array, label) ie Shape([self.Train, "Train"])
	"""
	npObj, label = lst; print(label + " shape: " +  str(npObj.shape))

def is_numeric(x):
	booll = (type(x) == float) or (type(x) == int)
	return(booll)

#seems useless, but we will see.
def class_copy(class_spec):
	CopyOfClass = type('esn_cv_copy', class_spec.__bases__, dict(class_spec.__dict__))
	return class_spec

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

class Reservoir:
    """
    a reservoir class that stores hyper-parameters specific to the reservoir, 
    as well as a csc matrix and the reservoir type.
    """
    def __init__(self, res_type, sparse_reserveroir_representation = None):
        self.res_type_ = res_type
        self.sparse_reserveroir_representation = sparse_reserveroir_representation

    def plot(self):
        print("Not implimented")

class InWeights:
    def __init__(self, in_weight_type, in_weights = None):
        self.in_weight_type = in_weight_type
        self.in_weights = in_weights

    def plot(self):
        print("Not implimented")
        
class ExperData:
    """
    Consider making this do the splitting automatically for based on the indices.

    f: frequency indices
    T: time indices
    """
    def __init__(self, dataset, f, T):
        # Observers_Train, Observers_Test, Target_Train, Target_Test,
        self.A_ = dataset
        self.set_shapes = {}
        self.sets = {}
        self.f_ = f
        self.T_  = T
        self.n_rows, self.n_cols = self.A_.shape 

    def add_data(self, time_indices, y_indices, name):
        
        ds = ExperDataSet(time_indices = time_indices, 
                          y_indices = y_indices, 
                          name = name,
                          dataset = self.A_.copy())
        #option2:
        #if name == "train"
        #    self.train = ExperDataSet(...)
        return ds
        
    def print_sets(self):
        printc("dataset: shape, " + str(self.set_shapes), 'blue')
    
    def create_datasets(self, resp_idx, obs_idx = None, split = None, train_time_idx = None, test_time_idx = None):
        self.obs_idx = obs_idx
        self.resp_idx = resp_idx
        if split:
	        train_len = int(self.n_rows * split)
	        test_len  = self.A_.shape[0] - train_len

	        #time indices
	        if not train_time_idx:
	            train_time_idx = list(range(train_len))
	            
	        if not test_time_idx:
	            test_time_idx  = list(range(train_len, self.n_rows))
	        assert len(train_time_idx) + len(test_time_idx) == self.A_.shape[0]
    
        #Target_Train
        self.sets["Target_Tr"] = self.add_data(time_indices = train_time_idx, y_indices = resp_idx, name = "Target_Tr")

        #Target_test
        self.sets["Target_Te"] = self.add_data(time_indices = test_time_idx, y_indices = resp_idx, name = "Target_Te")
        
        self.Target_Tr_, self.Target_Te_ = self.sets["Target_Tr"].data, self.sets["Target_Te"].data
        
        self.set_shapes = {"Target_Tr" : self.Target_Tr_.data.shape, 
                           "Target_Te": self.Target_Te_.data.shape}
        if obs_idx:
            #Obs_Train
            self.sets["Obs_Tr"] = self.add_data(time_indices = train_time_idx, y_indices = obs_idx, name = "Obs_Tr")

            #Obs_test
            self.sets["Obs_Te"] = self.add_data(time_indices = test_time_idx, y_indices = obs_idx, name = "Obs_Te")

            self.Obs_Tr_, self.Obs_Te_ = self.sets["Obs_Tr"].data, self.sets["Obs_Te"].data

            self.set_shapes = {"Obs_Tr" : self.Obs_Tr_.data.shape, "Obs_Te": self.Obs_Te_.data.shape}

        else:
            self.Obs_Tr_, self.Obs_Te_ = None, None



class ExperDataSet(ExperData):

    """
    Consider renaming this.
    Parameters:
        time_indices are the vertical indices that correspond to the time steps of the data.
        y_indices correspond to the variable perpendicular to time. For example, the frequencies
    """
    def __init__(self, time_indices, y_indices, name, dataset):
        self.time_indices_ = time_indices
        
        assert name in ["Target_Tr", "Target_Te", "Obs_Tr", "Obs_Te"]

        self.data = dataset.copy()

        #print(dataset.shape, "dataset shape")
        #print(self.time_indices_, "time_indices")

        self.data = dataset[np.array(self.time_indices_), :] 
        
        if y_indices:
        	
	        self.y_indices_ = y_indices
	        #print(self.y_indices_, "y_indices")
	        y_idx =  np.array(self.y_indices_)
	        if type(y_indices) == int:
	            self.data = self.data[:, y_idx].reshape(-1,1)
	        else:
	            self.data = self.data[:, y_idx]

        
        self.__shape = self.data.shape
        
    def print_indices(self):
        print("time_indices:", self.time_indices_)
        print("y_indices:", self.y_indices_)
        
    def print_matlab_indices(self):
        
        print("time_indices:", list(np.array(self.time_indices_ ) - 1))
        print("y_indices:", list(np.array(self.y_indices_ ) - 1))
    
    def print_shape(self):
        print("time_indices:", )

class Reservoir:
    """
    a reservoir class that stores hyper-parameters specific to the reservoir, 
    as well as a csc matrix and the reservoir type.
    """
    def __init__(self, res_type, sparse_reserveroir_representation = None):
        self.res_type_ = res_type
        self.sparse_reserveroir_representation = sparse_reserveroir_representation

    def plot(self):
        print("Not implimented")

class InWeights:
    def __init__(self, in_weight_type, in_weights = None):
        self.in_weight_type_ = in_weight_type
        self.in_weights = in_weights

    def plot(self):
        print("Not implimented")

class modelResult:
    def __init__(self, reservoir, input_weights, hyper_params, prediction, EchoStateExperiment_inputs, ip = False):
        
        if not ip:
            self.reservoir = reservoir
            self.input_weights = input_weights
            if hyper_params:
            	self.hyper_params = defaultdict(**hyper_params)
            if self.reservoir and self.input_weights:
            	self.name_ = self.reservoir.res_type_ + "_" + self.input_weights.in_weight_type_
        else:
            self.name_ = "interpolation"
        self.EchoStateExperiment_inputs = EchoStateExperiment_inputs 
        self.prediction = prediction
        
    def display_params(self):
        hyper_params = sorted(self.hyper_params.items())
        print(self.name_, "hyper parameters", hyper_params)
    
    def get_params(self):
        return dict(self.hyper_params)

    def get_residuals(self, ground_truth):
        return ground_truth - self.prediction



class ExperResult:
	"""
	The base class for an experiment result. This will be more general than RC_Result, containing things like the data.
	Parameters:
	    reservoir: a sparse matrix representation of the reservoir.
	    data: a numpy array (A) containing the entirity of the dataset.
	    
	"""
	def __init__(self, data, cv_inputs, get_observers_input):
		self.data = data
		self.cv_inputs = cv_inputs
		self.get_observers_input = get_observers_input
		self.model_results = {}

	def add_model_result(self, prediction, ground_truth, 
							reservoir = None, 
							input_weights = None, 
							hyper_params = None, 
							EchoStateExperiment_inputs = None,
							name = None,
						    ip = None):

		model_result = modelResult(reservoir = reservoir, 
						     input_weights = input_weights, 
						     hyper_params = hyper_params, 
						     prediction = prediction,
						     EchoStateExperiment_inputs = EchoStateExperiment_inputs,
						     ip = ip)
		if not ip:
			#ground_truth = self.data.Target_Te_)
			if reservoir:
				res_type = model_result.reservoir.res_type_
			if input_weights:
				input_weight_type = input_weights.in_weight_type_
				self.model_results[res_type +"_"+ input_weight_type] = model_result
			elif name:
				self.model_results[name] = model_result
		else:
			self.model_results["interpolation"] = model_result
	def rename_model(self, old_name, new_name):
		self.model_results[old_name].name_ = new_name
		self.model_results[new_name] = self.model_results[old_name]
		self.model_results = ifdel(self.model_results, old_name)
	def get_models(self):
		return list(self.model_results.keys())
        
	def display_experiment(self, log = True):
		pass 

	def get_model_result(self, model):
		return self.model_results[model]
        


class EchoStateExperiment:
	""" #Spectrogram class for training, testing, and splitting data for submission to reservoir nueral network code.
	
	Args: #TODO Incomplete
		bounds:
		size: a string in ["small", "medium", "publish"] that refer to different dataset sizes.
		file_path: a string that describes the directory where the data is located. (load from)
		out_path: where to save the data
		target_frequency: in Hz which frequency we want to target as the center of a block experiment or the only frequency in the case of a simple prediction.

	"""
	def __init__(self, size, 
				 file_path = "spectrogram_data/", target_frequency = None, out_path = None, obs_hz = None, 
				 target_hz = None, train_time_idx = None, test_time_idx = None, verbose = True,
				 smooth_bool = False, interpolation_method = "griddata-linear", prediction_type = "block",
				 librosa = False, spectrogram_path = None, flat = False, obs_freqs  = None,
				 target_freqs = None, spectrogram_type = None, k = None, obs_idx = None, resp_idx = None,
				 model = None, chop = None, input_weight_type = None, EchoStateExperiment_inputs = None
				 ):
		self.EchoStateExperiment_inputs =  EchoStateExperiment_inputs

		# Parameters



		self.size = size
		self.flat = flat
		self.spectrogram_type = spectrogram_type
		self.input_weight_type = input_weight_type 

		self.bounds = {"observer_bounds" : None, "response_bounds" : None} 
		self.esn_cv_spec = class_copy(EchoStateNetworkCV)
		self.esn_spec	= class_copy(EchoStateNetwork)
		self.file_path = file_path + self.size + "/"
		self.interpolation_method = interpolation_method
		self.json2be = {}
		self.librosa = librosa

		self.out_path = out_path
		self.prediction_type = prediction_type
		self.smooth_bool = smooth_bool
		self.spectrogram_path = spectrogram_path
		
		self.verbose = verbose
		self.target_freqs = target_freqs
		self.obs_freqs = obs_freqs
		self.obs_idx = obs_idx
		self.resp_idx = resp_idx
		self.chop = None #0.01/2

		assert model in ["uniform", "random",  "exponential", "delay_line", "cyclic"]
		self.model = model

		if obs_freqs:
			self.target_frequency =  float(np.mean(target_freqs))
		else:
			self.target_frequency = target_frequency
		
		#these indices should be exact lists, not ranges.
		if train_time_idx:
			self.train_time_idx = train_time_idx
		if test_time_idx:
			self.test_time_idx  = test_time_idx

		if self.prediction_type == "column":
			self.target_frequency = 100

		#print(self.prediction_type)
		assert self.target_frequency, "you must enter a target frequency"
		assert is_numeric(self.target_frequency), "you must enter a numeric target frequency"
		assert size in ["small", "medium", "publish", "librosa"], "Please choose a size from ['small', 'medium', 'publish']"
		assert type(verbose) == bool, "verbose must be a boolean"
		
		#order dependent attributes:
		self.load_data()

		#This deals with obs_freqs or obs_hz. We need a different method if we already have the exact index.
		if self.prediction_type == "block" or self.prediction_type == "freqs":
			#if the observer index hasn't been initialized or if it equal to None do this:
			if not self.obs_idx:
				if obs_freqs:
					self.obs_idx  = [self.Freq2idx(freq) for freq in obs_freqs]
					self.resp_idx = [self.Freq2idx(freq) for freq in target_freqs]
					self.resp_idx = list(np.unique(np.array(self.resp_idx)))
					
					self.obs_idx = list(np.unique(np.array(self.obs_idx)))
					
					for i in self.resp_idx:
						if i in self.obs_idx:
							self.obs_idx.remove(i)

					

				if obs_hz and target_hz:
					assert is_numeric(obs_hz), "you must enter a numeric observer frequency range"
					assert is_numeric(target_hz), "you must enter a numeric target frequency range"
				if not target_freqs:
					#print("great success")
					self.hz2idx(obs_hz = obs_hz, target_hz = target_hz)

		## exact assumes you already have the obss_idx
		elif self.prediction_type == "exact":
			self.obs_idx = list(np.unique(np.array(self.obs_idx)))
			self.resp_idx = list(np.unique(np.array(self.resp_idx)))
			for i in self.resp_idx:
				if i in self.obs_idx:
					self.obs_idx.remove(i)

		
		self.horiz_display()
		self.k = k

	def build_distance_matrix(self, dual_lambda, verbose = False):
		"""	
		args:
		    resp is the response index (a list of integers associated with the target train/test time series 
		        (for example individual frequencies)
		    obs is the same for the observation time-series.
		Description:
			DistsToTarg stands for distance numpy array
		"""
		def calculate_distance_matrix(obs_idx):
			obs_idxx_arr = np.array(obs_idx)
			for i, resp_seq in enumerate(self.resp_idx):
				DistsToTarg = abs(resp_seq - obs_idxx_arr).reshape(1, -1)
				if i == 0:
					distance_np_ = DistsToTarg
				else:
					distance_np_ = np.concatenate([distance_np_, DistsToTarg], axis = 0)
			distance_np_ = distance_np_
			
			return(distance_np_)

		if not dual_lambda:

			self.distance_np = calculate_distance_matrix(self.obs_idx) 
		else:

			def split_lst(lst, scnd_lst):
			    
				lst = np.array(lst)
				breaka = np.mean(scnd_lst)
				scnd_arr = np.array(scnd_lst)
				lst1, lst2 = lst[lst < scnd_arr.min()], lst[lst > scnd_arr.max()]

				return list(lst1), list(lst2)

			obs_lsts = split_lst(self.obs_idx, self.resp_idx) #good!
			self.distance_np = [calculate_distance_matrix(obs_lst) for obs_lst in obs_lsts]
			

	def save_pickle(self, path, transform):
		self.getData2Save()
		save_path = path # + ".pickle"
		with open(save_path, 'wb') as handle:
			pickle.dump(transform, handle, protocol=pickle.HIGHEST_PROTOCOL)

	def hz2idx(self, obs_hz = None, target_hz = None, silent = True):
		""" This function acts as a helper function to simple_block and get_observers
		and is thus designed. It takes a desired hz amount and translates that to indices of the data.
		
		To do one frequency use Freq2idx.

		Args:
			obs_hz: the number of requested observer hertz
			target_hz: the number of requested target hertz
			silent: if False prints general information
		"""
		if not silent:
			print("RUNNING HZ2IDX")
		midpoint = self.target_frequency 
		height   = self.freq_axis_len 

		#items needed for the file name:
		self.obs_kHz, self.target_kHz = obs_hz / 1000, target_hz / 1000

		# spread vs total hz
		obs_spread, target_spread = obs_hz / 2, target_hz / 2

		### START hz2idx helper functions
		def my_sort(lst): #sorts a list,
			try:
				lst.sort()
				return(lst)
			except:
				return(lst)
			#list(np.sort(lst)))

		def drop_overlap(resp_lst, obs_lst):
			""" # helper function for hz2idx. drops overlap from observers.
			"""
			intersection_list = list(set.intersection(set(resp_lst), set(obs_lst)))
			for i in intersection_list:
				obs_lst.remove(i)

			return(obs_lst)

		def endpoints2list(lb, ub, obs_spread = obs_spread, height = height): # obs = False, 
			"""
			Args:
				[lb, ub] stand for [lowerbound, upperbound]
				obs_spread: how much the observer extends in each direction (frequency)
			"""

			f = np.array(self.f)

			def librosa_range(lb_, ub_):
				"""
				takes lower and upper bounds and returns the indices list. Potentially could make the else statement
				below obselete.
				"""
				bounds = my_sort([lb_, ub_])
				lb, ub = bounds
				lb_bool_vec, ub_bool_vec = (f >= lb), (f <= ub)
				and_vector = ub_bool_vec * lb_bool_vec

				freqs = f[and_vector]			#frequencies between bounds
				freq_idxs = np.where(and_vector)[0].tolist() #indices between bounds
				return(freq_idxs)

			resp_range = librosa_range(lb, ub)

			respLb, respUb = f[resp_range][0], f[resp_range][-1]
			obs_hLb, obs_hUb = respUb, respUb + obs_spread 
			obs_lLb, obs_lUb = respLb - obs_spread, respLb 

			obs_L, obs_H = librosa_range(obs_lLb, obs_lUb), librosa_range(obs_hLb, obs_hUb )

			#drop the lowest index of obs_H and the highest index of obs_L to avoid overlap with resp_idx
			obs_H = drop_overlap(resp_lst = resp_range, obs_lst = obs_H)
			obs_L = drop_overlap(resp_lst = resp_range, obs_lst = obs_L)

			ranges = (resp_range, obs_L, obs_H ) 

			final_ranges = []
			for range_ in ranges:
				assert len(np.unique(range_)) == len(range_)
				range_ = [int(idx) for idx in range_] #enforce integer type
				range_ = [height - i for i in range_] #Inversion
				range_ = my_sort(range_)
				final_ranges.append(range_)
			#hack

			return(ranges)
		### END helper functions

		# get the obs, response range endpoints
		resp_freq_Lb, resp_freq_Ub = [midpoint - target_spread, midpoint + target_spread]
		
		#This could be incorrect but I'm trying to protect the original block method.
		#respLb, respUb = [self.Freq2idx(midpoint - target_spread), self.Freq2idx(midpoint + target_spread)]

		#print("frequencies: " + str([ round(x,1) for x in self.f]))
		#print("bounds2convert: (" +str(midpoint - target_spread ) + ", " + str(midpoint + target_spread ) + str(")"))

		#current_location

		# Listify:
		resp_idx_Lst, obs_idx_Lst1, obs_idx_Lst2 = endpoints2list(resp_freq_Lb, resp_freq_Ub)

		def get_frequencies(idx_lst):
			freq_lst = [self.f[idx] for idx in idx_lst]
			return freq_lst

		for i, idx_lst in enumerate([resp_idx_Lst, obs_idx_Lst1, obs_idx_Lst2]):
			if not i:
				freq_lst = []
			freq_lst += [get_frequencies(idx_lst)]

		resp_freq_Lst, obs_freq_Lst1, obs_freq_Lst2 = freq_lst

		if not silent:
			print("resp_indexes : " + str(resp_idx_Lst))
			print("observer frequencies upper domain: " + str(resp_freq_Lst) + 
				  " , range: "+ str(abs(resp_Freq_Lst[0] - resp_freq_Lst[-1])) +" Hz\n")

			print("observer indexes lower domain: " + str(obs_idx_Lst1))
			print("observer frequencies lower domain: " + str(obs_freq_Lst1) + 
				  " , range: "+ str(abs(obs_Freq_Lst1[0] - obs_freq_Lst1[-1])) +" Hz\n")

			print("observer indexes upper domain: " + str(obs_idx_Lst2))
			print("observer frequencies upper domain: " + str(obs_freq_Lst2) + 
				  " , range: "+ str(abs(obs_Freq_Lst2[0] - obs_freq_Lst2[-1])) +" Hz\n")
		
		#if not self.librosa:
		#	assert obs_idx_Lst2 + resp_idx_Lst + obs_idx_Lst1 == list(range(obs_idx_Lst2[ 0 ], obs_idx_Lst1[ -1] + 1))

		dict2Return = {"obs_idx": obs_idx_Lst2 + obs_idx_Lst1, 
					   "resp_idx": resp_idx_Lst,
					   "obs_freq" : obs_freq_Lst1 + obs_freq_Lst2,
					   "resp_freq" : resp_freq_Lst}

		self.resp_obs_idx_dict = dict2Return

		self.obs_idx  = [int(i) for i in dict2Return["obs_idx"]]
		#print("OBS IDX: " + str(self.obs_idx))

		self.resp_idx = [int(i) for i in dict2Return["resp_idx"]]
		#print("Resp IDX: " + str(self.resp_idx))


	def smooth(self, sigma = 1):
		""" # gaussian smoothing
		"""
		from scipy.ndimage import gaussian_filter
		#from scipy.ndimage import gaussian_filter
		self.A = gaussian_filter( self.A, sigma = sigma)

	def load_data(self, smooth = False):
		"""Loads data.

		Args: 
			smooth: a gaussian filter
			
		"""
		
		if self.librosa:
			spectrogram_path = "./pickle_files/spectrogram_files/" + self.spectrogram_path + ".pickle"
			
			with open(spectrogram_path, 'rb') as handle:
				pickle_obj = pickle.load(handle)

			self.f = np.array(pickle_obj["transform"]["f"])
			self.f = self.f.reshape(-1,).tolist()

			#self.spectrogram_type = method[1]
			assert self.spectrogram_type in ["power", "db"]

			if self.spectrogram_type == "power":
				self.A_unnormalized = pickle_obj["transform"]["Xpow"]
			else:
				self.A_unnormalized = pickle_obj["transform"]["Xdb"]
		else:
			spect_files  = { "publish" : "_new", "small" : "_512" , "original" : "", "medium" : "_1024"}

			files2import = [self.file_path  + i + spect_files[self.size] for i in ("T", "f", "Intensity") ]
			
			data_lst = []
			for i in files2import:
				data_lst.append(loadmat(i))

			self.T, self.f, self.A = data_lst #TODO rename T, f and A (A should be 'spectrogram' or 'dataset')

			self.A_unnormalized = self.A['M'].T.copy()

			#preprocessing
			self.T = self.T['T']
			self.T = np.transpose(self.T)
			self.f = self.f['f'].reshape(-1,).tolist()

		#copy the matrix
		self.A = self.A_unnormalized.copy()

		if self.chop:
			n_timeseries = self.A.shape[1]
			n_timeseries = int(self.chop * n_timeseries)
			self.f = self.f[ : n_timeseries]
			self.A = self.A[ : , : n_timeseries]

		#normalize the matrix
		self.A = (self.A - np.mean(self.A)) / np.std(self.A)

		#gaussian smoothing
		if self.smooth_bool:
			self.smooth()

		self.max_freq = int(np.max(self.f))
		self.Freq2idx(self.target_frequency, init = True)

		self.freq_axis_len = self.A.shape[0]
		self.time_axis_len = self.A.shape[1]

		str2print = ""
		if self.verbose:
			for file_name_ in files2import:
				str2print += "successfully loaded: " + file_name_ + ".mat, "
			#print("maximum frequency: " + str(self.max_freq))
			#print("dataset shape: " + str(self.A.shape))

		self.key_freq_idxs = {}
		for i in (2000, 4000, 8000):
			height = self.A.shape[0]
			self.Freq2idx(i)
			self.key_freq_idxs[i] = height - self.targetIdx


	def olab_display(self, axis, return_index = False):
		"""
		#TODO reconsider renaming vert_display
		Plot a version of the data where time is along the x axis, designed to show RPI lab
		"""
		oA = np.rot90(self.A.copy(), k = 3)#3, axes = (0, 1))
		#oA stands for other lab A
		oA = pd.DataFrame(oA).copy()
		
		oA.index = self.freq_idx
		yticks = list( range( 0, self.max_freq, 1000))
		y_ticks = [ int(i) for i in yticks]
		my_heat = sns.heatmap(oA, center=0, cmap=sns.color_palette("CMRmap"), yticklabels = self.A.shape[0]//10, ax = axis)
		#, cmap = sns.color_palette("RdBu_r", 7))
		axis.set_ylabel('Frequency (Hz)')#,rotation=0)
		axis.set_xlabel('time')
		my_heat.invert_yaxis()
		my_heat.invert_xaxis()
		plt.yticks(rotation=0)
		if return_index:
			return(freq_idx)


	def Freq2idx(self, val, init = False):
		"""
		Translates a desired target frequency into a desired index
		"""
		freq_spec = min(range(len(self.f)), key=lambda i: abs(self.f[i] - val))
		assert type(init) == bool, "init must be a bool"
		if init == False:
			return(freq_spec)
		else:
			self.targetIdx = freq_spec
	
	#TODO: horizontal display
	def horiz_display(self, plot = False):
		assert type(plot) == bool, "plot must be a bool"
		A_pd = pd.DataFrame(self.A)
		A_pd.columns = self.f
		if plot:
			fig, ax = plt.subplots(1,1, figsize = (6,4))
			my_heat= sns.heatmap(A_pd,  center=0, cmap=sns.color_palette("CMRmap"), ax = ax)
			ax.set_xlabel('Frequency (Hz)')
			ax.set_ylabel('time')
		self.A = A_pd.values

	#TODO plot the data



	def build_pd(self, np_, n_series):
		series_len = np_.shape[0]
		for i in range(n_series): 
			id_np =  np.zeros((series_len, 1)).reshape(-1, 1) + i
			series_spec = np_[:, i].reshape(-1, 1)
			t = np.array( list( range( series_len))).reshape(-1, 1)
			pd_spec = np.concatenate( [ t, series_spec, id_np], axis = 1)
			pd_spec = pd.DataFrame(pd_spec)
			pd_spec.columns = ["t", "x", "id"]
			if i == 0:
				df = pd_spec 
			else:
				df = pd.concat([df, pd_spec], axis = 0)
		return(df)


	def plot_timeseries(self, 
						titl = "ESN ", 
						series2plot = 0, 
						method = None, 
						label_loc = (0., 0.)): #prediction_, train, test, 
		'''
		This function makes three plots:
			the prediction, the residual, the loss.
		It was built for single predictions, but needs to be upgraded to deal with multiple output.
		We need to show: average residual, average loss.
		'''
		prediction_ = self.prediction
		train = self.Train
		test  = self.Test

		full_dat = np.concatenate([train, test], axis = 0); full_dat_avg = np.mean(full_dat, axis = 1)
		n_series, series_len = test.shape[1], test.shape[0]
		assert method in ["all", "single", "avg"], "Please choose a method: avg, all, or single"
		#assert method != "all", "Not yet implimented #TODO"
		
		if method == "single":
			label_loc = (0.02, 0.65)
		
		#key indexes
		trainlen, testlen, pred_shape = train.shape[0], test.shape[0], prediction_.shape[0]
		
		if method == "single":
			if n_series > 1:
				print("There are " + str(n_series) + " time series, you selected time series " 
					+ str(series2plot + 1))
			
			# avoid choosing all of the columns. subset by the selected time series.
			train, test, prediction = train[:, series2plot], test[:, series2plot], prediction_[:, series2plot]
			
			
			# set up dataframe
			xTrTarg_pd = pd.DataFrame(test)
			t = pd.DataFrame(list(range(len(xTrTarg_pd))))
			
			# append time
			Target_pd = pd.concat([xTrTarg_pd, t], axis = 1)
			Target_pd.columns = ["x", "t"]
			
			 #calculate the residual
			resid = test.reshape(-1,)[:pred_shape] - prediction.reshape(-1,) #pred_shape[0]
			
			rmse_spec =  str(round(myMSE(prediction, test), 5))
			full_dat = np.concatenate([train, test], axis = 0)
			
		elif method == "avg":
			rmse_spec =  str(round(nrmse(prediction_, test), 5))
			prediction = prediction_.copy().copy()
			
			def collapse(array):
				return(np.mean(array, axis = 1))

			vals = []
			#y - yhat
			resid_np = test - prediction_
			
			for i in [train, test, prediction_, resid_np]:
				vals.append(collapse(i))
				
			train, test, prediction_avg, resid = vals
			#return(prediction)
		else: ##############################################################################################
			#TODO make a loop and finish this, hopefully pretty colors.
			
			rmse_spec =  str(round(nrmse(prediction_, test), 5))
			
			pd_names = ["Lines", "prediction", "resid"]
			pd_datasets = [ full_dat, prediction_, test - prediction_]
			rez = {}
			
			for i in range(3):
				# TODO functionalize this to streamline the other plots.
				name_spec = pd_names[i]
				dataset_spec = pd_datasets[i]
				rez[name_spec] = build_pd(dataset_spec, n_series)
				
			Lines_pd, resid_pd, prediction_pd = rez["Lines"], np.abs(rez["resid"]), rez["prediction"]
			#display(Lines_pd) #np.zeros((4,1))
		
		####### labels
		if method in ["single"]:	
			plot_titles = [ titl + "__: Prediction vs Ground Truth, rmse_: " + rmse_spec,
						   titl + "__: Prediction Residual",
						   titl + "__: Prediction Loss"]
			plot_labels = [
				["Ground Truth","prediction"]
			]
		elif method == "avg":
			plot_titles = [titl + "__: Avg Prediction vs Avg Ground Truth, total rmse_: " + rmse_spec,
						   titl + "__: Avg Prediction Residual",
						   titl + "__: Avg Prediction Loss"]
			plot_labels = [
				[ "", "Avg Ground Truth", "avg. prediction"]
			]
		elif method == "all":
			plot_titles = [titl + "__: Visualization of Time series to Predict, rmse_: " + rmse_spec,
						   titl + "__: Prediction Residuals", titl + "__: Prediction Loss"
						  ]
		
		### [plotting]	
		
		
		
		#display(Target_pd)
		fig, ax = plt.subplots(3, 1, figsize=(16,10))
		
		i = 0 # plot marker
		j = 0 # subplot line marker
		
		######################################################################## i. (avg.) prediction plot
		if method in ["single", "avg"]:
			
			if method == "single": col, alph = "cyan", 0.5,
			else: col, alph = "grey", 0.3
			
			### ground truth
			ax[i].plot(range(full_dat.shape[0]), full_dat,'k', label=plot_labels[i][j],
					  color = col, linewidth = 1, alpha = alph); j+=1
			
			if method == "avg":
				ax[i].plot(range(full_dat.shape[0]), full_dat_avg,'k', label=plot_labels[i][j],
					  color = "cyan", linewidth = 1, alpha = 0.8); j+=1
				# ground truth style
				ax[i].plot(range(full_dat.shape[0]), full_dat_avg,'k', color = "blue", linewidth = 0.5, alpha = 0.4)
			else:
				# ground truth style
				ax[i].plot(range(full_dat.shape[0]), full_dat,'k', color = "blue", linewidth = 0.5, alpha = 0.4)
			
			
			### prediction
			#pred style, pred
			if method == "single":
				ax[i].plot(range(trainlen,trainlen+testlen), prediction,'k',
						 color = "white",  linewidth = 1.75, alpha = .4)
				ax[i].plot(range(trainlen,trainlen+testlen), prediction,'k',
						 color = "red",  linewidth = 1.75, alpha = .3)
				ax[i].plot(range(trainlen,trainlen+testlen),prediction,'k',
						 label=plot_labels[i][j], color = "magenta",  linewidth = 0.5, alpha = 1); j+=1
			else: #potentially apply this to the all plot as well. Maybe only have two methods.
				ax[i].plot(range(trainlen,trainlen+testlen), prediction,'k',
						 color = "pink",  linewidth = 1.75, alpha = .35)
				ax[i].plot(range(trainlen,trainlen+testlen), prediction_avg,'k',
						 color = "red",  linewidth = 1.75, alpha = .4, label = "prediction avg")		   #first plot labels		   ax[i].set_title(plot_titles[i])		   ax[i].legend(loc=label_loc)		   i+=1; j = 0	   else:		   sns.lineplot( x = "t", y = "x", hue = "id", ax = ax[i], 						data = Lines_pd, alpha = 0.5,						palette = sns.color_palette("hls", n_series))		   ax[i].set_title(plot_titles[i])		   i+=1	   	   if method in ["single", "avg"]:		   ######################################################################## ii. Residual plot		   ax[i].plot(range(0,trainlen),np.zeros(trainlen),'k',					label="", color = "black", alpha = 0.5)		   ax[i].plot(range(trainlen, trainlen + testlen), resid.reshape(-1,),'k',					color = "orange", alpha = 0.5)		   # second plot labels		   #ax[1].legend(loc=(0.61, 1.1))		   ax[i].set_title(plot_titles[i])		   i+=1	   else:		   resid_pd_mn = resid_pd.pivot(index = "t", 										columns = "id", 										values = "x"); resid_pd_mn = resid_pd_mn.mean(axis = 1)	   		   sns.lineplot( x = "t", y = "x", hue = "id", ax = ax[i], data = resid_pd, alpha = 0.35, label = None)		   for j in range(n_series):			   ax[i].lines[j].set_linestyle((0, (3, 1, 1, 1, 1, 1)))#"dashdot")		   		   sns.lineplot(ax = ax[i], data = resid_pd_mn, alpha = 0.9, color = "r",						 label = "mean residual")		   		   ax[i].set_title(plot_titles[i])		   i+=1	   ####################################################################### iii. Loss plot	   if method in ["single", "avg"]:		   		   ax[i].plot(range(0,trainlen),np.zeros(trainlen),'k',					label="", color = "black", alpha = 0.5)		   ax[i].plot(range(trainlen,trainlen+testlen),resid.reshape(-1,)**2,'k',					color = "r", alpha = 0.5)		   # second plot labels		   #ax[2].legend(loc=(0.61, 1.1))		   ax[i].set_title(plot_titles[i])		   	   elif method == "all":		   # create the loss dataframe		   loss_pd = resid_pd.copy(); 		   vals =  loss_pd['x'].copy().copy(); loss_pd['x'] = vals **2		   		   loss_pd_mn = loss_pd.pivot(index = "t", 										columns = "id", 										values = "x"); loss_pd_mn = loss_pd_mn.mean(axis = 1)	   		   sns.lineplot( x = "t", y = "x", hue = "id", ax = ax[i], data = loss_pd, alpha = 0.35, label = None)		   for j in range(n_series):			   ax[i].lines[j].set_linestyle((0, (3, 1, 1, 1, 1, 1)))#"dashdot")		   		   sns.lineplot(ax = ax[i], data =loss_pd_mn, alpha = 0.9, color = "magenta",						 label = "mean loss")		   		   ax[i].set_title(plot_titles[i])		   i+=1	   plt.subplots_adjust(hspace=0.5)
		plt.show()

	def diff(self, first, second):
		second = set(second)
		return [item for item in first if item not in second]

	def myMSE(prediction,target):
		return np.sqrt(np.mean((prediction.flatten() - target.flatten() )**2))

	

	# validation version
	def get_observers(self, 
					  missing = None,  # missing = self.key_freq_idxs[2000], 
					  aspect = 6,
					  method  = "random", 
					  num_observers = 20,
					  plot_split = False,
					  split = 0.2,
					  get_observers_input = None
					  ): 
		"""
		arguments:
			aspect: affect the size of the returned plot.
			dataset: obvious
			method: 
				(+) random 
				(+) equal #similar to barcode, equal spacing, with k missing block. Low priority.
				(+) block
				(+) barcode #TODO block but with gaps between observers.
					# I think this will show that you don't really need every line of the data to get similar accuracy
			
			missing: either 
				(+) any integer:  (standing for column of the spectrogram) or 
				(+) "all" : which stands for all of the remaining target series.
			num_observers: the number of observers that you want if you choose the "random" method.
			observer_range: if you select the "block" opion
		"""
		#preprocessing:
		self.get_observers_input = get_observers_input
		k = self.k
		dataset, freq_idx  = self.A,  self.f
		n_rows, n_cols = dataset.shape[0], dataset.shape[1]
		train_len = int(n_rows * split)
		test_len =  n_rows - train_len
		col_idx = list(range(n_cols))

		self.split  = split
		self.method = method
		self.aspect = aspect

		experData = ExperData(self.A, f = self.f, T = self.T)
		
		
		#remove the response column which we are trying to use for inpainting
		if method == "random":
			col_idx.remove(missing)
			self.obs_idx = np.random.choice(col_idx, num_observers, replace = False)
			self.resp_idx = [missing]
			
		elif method == "eq":
			print("equal spacing")
			print("NOT YET IMPLIMENTED")
			
		elif method == "all":
			self.obs_idx = np.random.choice( col_idx, num_observers, replace = False)
			self.resp_idx  = diff( col_idx, self.obs_idx.tolist())
		
		### BLOCK: this is oldschool and super-annoying: you have to specify indices.
		elif method == "block":
			"""
			This method either blocks observers and/or the response area.
			"""
			print("you selected the block method")

			if not self.resp_idx:
				self.resp_idx  = [missing]

			###Cool vestigal randomization.
			#if observer_range == None:
			#	col_idx.remove( missing)
			#	obs_idx = np.sort( np.random.choice( col_idx, num_observers, replace = False))
			

		elif method == "freq":
			"""
			The newest method, the only one we care to have survive because it is not based on indices but rather desired Hz.
			This method is just like simple_block but upgraded to take in only frequencies by using the helper function hz2freq which must
			be called first.
			"""
			
			assert self.obs_idx, "oops, your observer index cannot be None, first run hz2idx helper function"
			assert self.resp_idx, "oops, your response index cannot be None"

			#response = dataset[ : , self.resp_idx].reshape( -1, len( self.resp_idx))

		elif method == "exact":
			"""
			Exact indices.
			"""
			
			self.exact = True

		if self.prediction_type == "column":
			if not k or k == 1:
				self.obs_idx, self.resp_idx  = [], range(self.A.shape[1])
			elif k != 1:
				assert 1 == 0, "k is currently deactivated in the get_observers function."
			#else:
			#	self.obs_idx, self.resp_idx  = [], range(0, self.A.shape[1], k)

			self.target_kHz = "all"
			self.obs_kHz	= 0.0	


		assert method in ["freq", "exact"], "at this time only use the 'freq' method for cluster, 'exact' for analysis"

		if self.prediction_type != "column":

			# PARTITION THE DATA
			print("split", split)
			experData.create_datasets(resp_idx = self.resp_idx, obs_idx = self.obs_idx, split = split)
			#self.Train, self.Test = experData.Obs_Tr, experData.Obs_Te #observers[ :train_len, : ], observers[ train_len:, : ]
			#self.xTr, self.xTe    = experData.Target_Tr, experData.Target_Te #response[ :train_len, : ], response[ train_len:, : ]
		else:
			experData.create_datasets(resp_idx = self.resp_idx,
									  train_time_idx = self.train_time_idx, 
									  test_time_idx = self.test_time_idx)
			
		cool_k = """
		elif self.prediction_type == "column":
			#no observers so assign empty arrays
			

			#self.xTr = dataset[ self.train_time_idx, : ]
			#self.xTe = dataset[ self.test_time_idx , : ]

			
			if k: # calculating 1 in k observers:
				n_obs = A_shape_0 // k

				#rand_start = np.random.randint(k) <-- upgrade to this later.
				rand_start = 0 
				keep_index = list(range(rand_start, A_shape_0, k))

				#we are slimming down the inputs to reduce computational complexity.
				self.Train = self.Train[:, keep_index]
				self.Test  = self.Test[:, keep_index]"""
		self.Data = experData
		self.Train, self.Test = experData.Obs_Tr_, experData.Obs_Te_ 
		self.xTr, self.xTe    = experData.Target_Tr_, experData.Target_Te_

		experData.print_sets()
		
		#print("RESP IDX", self.resp_idx)
		#print("OBS IDX", self.obs_idx)

		### Visualize the train test split and the observers
		if plot_split:
			red, yellow, blue, black = [255, 0, 0], [255, 255, 0], [0, 255, 255], [0, 0, 0]
			orange, green, white = [255, 165, 0], [ 0, 128, 0], [255, 255, 255]

			#preprocess:
			split_img = np.full(( n_rows, n_cols, 3), black)

			# assign observer lines
			for i in self.obs_idx:
				split_img[ : , i] = np.full(( 1, n_rows, 3), yellow)

			if self.prediction_type != "column":
				# assign target area
				for i in response_idx:
					split_img[ :train_len, i] = np.full(( 1, train_len, 3), blue)
					split_img[ train_len:, i] = np.full(( 1, test_len,  3), red)
			else:
				for i in self.resp_idx:
					split_img[ self.train_time_idx, i ] = np.full(( 1, train_len, 3), blue)
					split_img[ self.test_time_idx,  i ] = np.full(( 1, test_len,  3), red)

			legend_elements = [Patch(facecolor='cyan', edgecolor='blue', label='Train'),
						   	   Patch(facecolor='red', edgecolor='red', label='Test'),
							   Patch(facecolor='yellow', edgecolor='orange', label='Observers')]
			
			
			# Create the figure
			fig, ax = plt.subplots( 1, 2, figsize = ( 12, 6))
			ax = ax.flatten()
			
			solid_color_np =  np.rot90(split_img, k = 1, axes = (0, 1))
			#np.transpose(split_img, axes = (1,2,0))

			
			
			#solid_color_pd.index = freq_idx
			
			# The legend:
			#https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
			
			
			##################################### START plots
			
			#++++++++++++++++++++++++++++++++++++ plot 1: sns heatmap on the right
			self.olab_display(ax[1])
			ax[1].set_title("spectrogram Data")
			
			# retrieve labels to share with plot 0
			# We need to retrieve the labels now.
			plt.sca(ax[1])
			locs, labels = plt.yticks()
			freq_labels = np.array([int(label.get_text()) for label in labels])
			
			#++++++++++++++++++++++++++++++++++++ plot 0: diagram showing training, test splits and observers. LHS
			ax[0].set_title("Dataset Split Visualization")
			ax[0].imshow(solid_color_np, aspect = aspect)
			
			### fixing labels on plot 0, involved!
			# label axes, legend
			ax[0].set_ylabel('Frequency (Hz)'); ax[0].set_xlabel('time')
			ax[0].legend(handles=legend_elements, loc='lower right')
			
			#now calculate the new positions
			max_idx = solid_color_np.shape[0]
			
			#new positions
			new_p = (freq_labels/self.max_freq) * max_idx 
			adjustment = max_idx - np.max(new_p); new_p += adjustment -1; new_p  = np.flip(new_p)
			plt.sca(ax[0]); plt.yticks(ticks = new_p, labels = freq_labels)
			###
			
			plt.show()
			
			##################################### END plots
			

		self.runInterpolation(k = k)

		#assert self.xTr.shape[1] == self.xTe.shape[1], "something is broken, xTr and xTe should have the same column dimension"
		
		
		self.outfile = "experiment_results/" 


		self.outfile += self.size

		self.outfile += "/split_" + str(split)  +"/"

		from datetime import datetime

		# datetime object containing current date and time
		now = datetime.now()
		 
		#print("now =", now)

		# dd/mm/YY H:M:S
		dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")

		self.outfile += dt_string + "__"

		if self.prediction_type == "column":
			self.outfile += "column_"
		elif self.prediction_type == "block":
			self.outfile += "block_"

		if self.model == "delay_line":
			self.outfile += "DL"
		elif self.model == "cyclic":
			self.outfile += "cyclic"
		elif self.model == "random":
			self.outfile += "random"

		if self.input_weight_type == "uniform":
			self.outfile += "_unif_W_"
		elif self.input_weight_type == "exponential":
			self.outfile += "_expo_W_"

		if self.method == "freq":
			ctr = int(np.mean([int(self.f[idx]) for idx in self.resp_idx]))
			self.outfile += "targetHz_ctr:_" + str(ctr)
			self.outfile += "targetKhz:_" + str(self.target_kHz) + "__obskHz:_" + str(self.obs_kHz)
		elif self.method == "exact":
			self.outfile += "N_Targidx_" + str(len(self.resp_idx)) 
			self.outfile += "N_Obsidx_" + str(len(self.obs_idx))

		#print("OUTFILE: " + str(self.outfile))


	def getData2Save(self): 
		'''
		Save the data
		current issue: how do we initialize this function properly?
		'''

		def jsonMerge(new_dict):
			self.json2be = Merge(self.json2be, new_dict)


		reservoir_  = Reservoir(res_type = self.model, sparse_reserveroir_representation = self.weights)
		in_weights_ = InWeights(in_weight_type = self.input_weight_type, in_weights = self.in_weights)

		if self.json2be == {}:
			print("initialiazing json2be")

			self.exper_result = ExperResult(data = self.Data, 
									   		cv_inputs = self.cv_args, 
									   		get_observers_input = self.get_observers_input)

			#add interpolation result.
			self.exper_result.add_model_result(prediction = self.ip_res["prediction"],  #fix later
											   ground_truth = self.xTe,
											   ip = self.ip_res["method"],
											   EchoStateExperiment_inputs = self.EchoStateExperiment_inputs)

			### save the result of 
			#self.runInterpolation() 
			#ip_pred = {"interpolation" : self.ip_res["prediction"]}
			#ip_nrmse = {"interpolation" : self.ip_res["nrmse"]}
			#jsonMerge({"prediction" : ip_pred})
			#jsonMerge({"nrmse" : ip_nrmse})
			#jsonMerge({"best arguments" : {}})

		self.exper_result.add_model_result(reservoir = reservoir_, 
										   input_weights = in_weights_, 
										   hyper_params = self.best_arguments, 
										   prediction = self.prediction,  #fix later
										   ground_truth = self.xTe,
										   EchoStateExperiment_inputs = self.EchoStateExperiment_inputs)

		self.json2be = {"exper_result" : self.exper_result}

		err_msg = "YOU NEED TO CALL THIS FUNCTION LATER "
		"""
		if not self.librosa: 
			# 1) Here stored are the inputs to 
			self.json2be["experiment_inputs"] = { #this should be built into the initialization to avoid errors.
				 "size" : self.size, 
				 "target_frequency" : int(self.target_frequency),
				 "verbose" : self.verbose,
				 }
			try:
			   json2be["experiment_inputs"]["obs_hz"] = float(self.obs_kHz)	* 1000
			   json2be["experiment_inputs"]["target_hz"] = float(self.target_kHz) * 1000
			except:
				print("")
			try:
			   json2be["experiment_inputs"]["target_freqs"] = self.target_freqs
			   json2be["experiment_inputs"]["obs_freqs"] = self.obs_freqs
			except:
				print("")
		"""

		"""
			self.json2be["experiment_inputs"] = {
				 "size" : self.size, 
				 "target_frequency" : int(self.target_frequency),
				 "obs_idx" : self.obs_idx,
				 "target_idx" : self.resp_idx,
				 "verbose" : self.verbose,
				 "prediction_type" : self.prediction_type
				 }
		self.json2be["get_observer_inputs"] = {
				"method" : self.method,
				"split" : self.split,
				"aspect" : float(self.aspect)
			}
		"""
		"""						 
		# 2) saving the optimized hyper-parameters, nrmse

		try:
			self.best_arguments
		except NameError:
			err_msg + "MISSING BEST ARGUMENTS< SERIOUS ERROR"

		args2export = self.best_arguments

		if self.input_weight_type == "exponential":
			assert self.esn_cv.input_weight_type == "exponential"
		elif self.input_weight_type == "uniform":
			args2export = ifdel(args2export, "llambda")
			args2export = ifdel(args2export, "llambda2")
		try:
			model_key = self.model + "_" + self.input_weight_type
			self.json2be["prediction"]= Merge(self.json2be["prediction"], { model_key : pred}) #Merge(self.json2be["prediction"], )
			self.json2be["nrmse"][model_key] = nrmse(pred, self.xTe, columnwise = False)
		except:
			print("object doesn't have the prediction attribute.")
		self.json2be["best arguments"] = Merge(self.json2be["best arguments"], {model_key : args2export}) 
		"""

	
	def RC_CV(self, cv_args, model, input_weight_type, hybrid_llambda_bounds = (-5, 1)): #TODO: change exp to 
		"""
		example bounds:
		bounds = {
			'llambda' : (-12, 1), 
			'connectivity': 0.5888436553555889, #(-3, 0)
			'n_nodes': (100, 1500),
			'spectral_radius': (0.05, 0.99),
			'regularization': (-12, 1),

			all are log scale except  spectral radius and n_nodes
		}
		example cv args:

		cv_args = {
			bounds : bounds,
			initial_samples : 100,
			subsequence_length : 250, #150 for 500
			eps : 1e-5,
			cv_samples : 8, 
			max_iterations : 1000, 
			scoring_method : 'tanh',
			exp_weights : False,
		}
		#esn_cv_spec equivalent: EchoStateNetworkCV
		"""
		if "esn_feedback" in cv_args.keys():
			self.feedback = cv_args["esn_feedback"]
		self.model = model
		self.input_weight_type = input_weight_type

		assert self.model in ["random", "delay_line", "cyclic"], self.model + " model not yet implimented"

		input_err_msg = " input weight type not yet implimented"
		assert self.input_weight_type in ["exponential", "uniform"], self.input_weight_type + input_err_msg

		predetermined_args = {"model_type" : self.model, "input_weight_type" : self.input_weight_type}

		if self.input_weight_type == "uniform":
			cv_args["bounds"] = ifdel(cv_args["bounds"], "llambda")
			cv_args["bounds"] = ifdel(cv_args["bounds"], "llambda2")
			cv_args["bounds"] = ifdel(cv_args["bounds"], "noise")

		### hacky intervention:
		if self.prediction_type != "column":
			predetermined_args = { 
			    **predetermined_args,
				'obs_index' : self.obs_idx,
				'target_index' : self.resp_idx
			}

		dual_lambda = True if "llambda2" in cv_args["bounds"] else False
			
		self.build_distance_matrix(dual_lambda = dual_lambda)
		

		input_dict = { **cv_args, 
					   **predetermined_args,
					   "Distance_matrix" : self.distance_np,
					   }
		self.cv_args = cv_args

		# subclass assignment: EchoStateNetworkCV
		self.esn_cv = self.esn_cv_spec(**input_dict)

		if self.model in ["random", "delay_line", "cyclic"]:
			if self.input_weight_type in ["uniform", "exponential"]:
				print(self.model, "rc cv with ", input_weight_type, " weights set, ready to train ")

		if self.prediction_type == "column":
			if self.feedback == True:
				self.esn_cv.feedback = True
				printc("training with feedback", "fail")
			else:
				self.esn_cv.feedback = False
				printc("training without feedback", "green")
			#printc("ATTEMPTING TO OPTIMIZE", "fail")
			#print(self.esn_cv)
			self.best_arguments =  self.esn_cv.optimize(x = None, y = self.xTr)
			
		else:
			self.feedback = False
			printc("ATTEMPTING TO OPTIMIZE", "fail")
			self.best_arguments =  self.esn_cv.optimize(x = self.Train, y = self.xTr) 
		self.best_arguments['feedback'] = self.feedback

		print("Bayesian Optimization complete. Now running saving data, getting prediction etc. ")
		print(input_dict)
		print(cv_args)
		printc("Training " + self.model + "_" + self.input_weight_type + " Reservoir", 'fail')
		
		self.esn = self.esn_spec(**self.best_arguments,
								 obs_idx  = self.obs_idx,
								 resp_idx = self.resp_idx, 
								 model_type = self.model,
								 input_weight_type = self.input_weight_type
								 )

		print(self.best_arguments)
		def my_predict(test, n_steps = None):
			if not n_steps:
				n_steps = test.shape[0]
			return self.esn.predict(n_steps, x = test[:n_steps,:])

		

		if self.prediction_type == "column":
			print()
			self.esn = self.esn_spec(**self.best_arguments,
								 obs_idx  = self.obs_idx,
								 resp_idx = self.resp_idx, 
								 model_type = self.model,
								 input_weight_type = self.input_weight_type
								 )
			nrmses = []
			for i in range(20):
				self.esn.train(x = None, y = self.xTr)
				self.weights = self.esn.weights
				self.in_weights = self.esn.in_weights
				self.prediction = self.esn.predict(x = None, n_steps = self.xTe.shape[0])
				nrmse_ = nrmse(self.xTe,self.prediction)
				nrmses.append(nrmse_)
			printc("nrmse: " + str( np.mean(nrmses)), 'cyan')
			
		else:
			self.esn.train(x = self.Train, y = self.xTr)
			self.weights = self.esn.weights
			self.in_weights = self.esn.in_weights
			self.prediction = my_predict(self.Test)
			rmse_ = nrmse(self.xTe,self.prediction)
			#	nrmses.append(nrmse_)
			printc("nrmse: " + str(rmse_), 'cyan')
		

		

		self.save_json()
		print("\n \n rc cv data saved @ : " + self.outfile +".pickle")


	def already_trained(self, best_args, model):
		
		self.best_arguments = best_args
		if best_args:
			extra_args = {}
			if self.model in ["delay_line", "cyclic"]:
				extra_args = {**extra_args, "activation_function" : "sin_sq"}
			
			best_args = {**best_args, **extra_args}

			if model in ["uniform", "exponential"]:
				self.input_weight_type = model
				self.model_type = "random"

			self.esn = self.esn_spec(**best_args,
									 obs_idx  = self.obs_idx,
									 resp_idx = self.resp_idx,
									 model_type = self.model_type,
									 input_weight_type = self.input_weight_type)

			def my_predict(test, n_steps = None):
				if not n_steps:
					n_steps = test.shape[0]
				return self.esn.predict(n_steps, x = test[ :n_steps, :])

			printc("best_args " + str(best_args), 'fail')
			if "feedback" in best_args:
				if best_args["feedback"] == True:
					self.esn.train(x = None, y = self.xTr)
					self.prediction = self.esn.predict(n_steps = self.xTe.shape[0])
				else:
					print("pure prediction with old data, no feedback not handled yet")
					self.esn.train(x = self.Train, y = self.xTr)
					self.prediction = my_predict(self.Test)
			else:
				self.esn.train(x = self.Train, y = self.xTr)
				self.prediction = my_predict(self.Test)
		else:
			"at least one network not trained successfully"

	def save_json(self):
		
		self.getData2Save()
		if self.librosa: # or self.prediction_type == "column"

			#build_string_lst = [self.spectrogram_path,"/",self.spectrogram_type,"/"

			# flat or NOT
			#TODO
			flat_str = "untouched" if not self.flat else "flat"
			librosa_outfile += str(flat_str) + "/"

			#split
			librosa_outfile += "split_"  + str(self.split) + "/"
			librosa_outfile += "type_" + str(self.prediction_type)

			librosa_outfile += "tf_" + str(self.target_frequency)

			#librosa_outfile = build_string("./pickle_files/results/", build_string_lst)
			librosa_outfile = "./pickle_files/results/" + self.spectrogram_path +"/" 
			# spectrogram type
			librosa_outfile += self.spectrogram_type + "/"

			# flat or NOT
			#TODO
			flat_str = "untouched" if not self.flat else "flat"
			librosa_outfile += str(flat_str) + "/"

			#split
			librosa_outfile += "split_"  + str(self.split) + "/"
			librosa_outfile += "type_" + str(self.prediction_type)

			librosa_outfile += "tf_" + str(self.target_frequency)
			if self.exact:
				librosa_outfile += "__obsNIdx_"  + str(len(self.obs_idx))
				librosa_outfile += "__targNIdx_" + str(len(self.resp_idx))
			else:
				librosa_outfile += "__obsHz_"  + str(self.obs_kHz)
				librosa_outfile += "__targHz_" + str(self.target_kHz)
			
			librosa_outfile += ".pickle"
			self.save_pickle(path = librosa_outfile, transform = self.json2be)
			print("librosa outfile: " + str(librosa_outfile))

		else:
			# For a host of reasons I'm retiring the old saving data method.
			new_file = self.outfile
			new_file += ".pickle"

			self.save_pickle(path = new_file, transform = self.json2be)

			#with open(new_file, "w") as outfile:
			#	data = json.dump(self.json2be, outfile)

	def rbf_add_point(self, point_tuple, test_set = False):

		x, y = point_tuple
		if test_set:
			self.xs_unknown  += [x]
			self.ys_unknown  += [y]
		else:
			self.xs_known += [x]
			self.ys_known += [y]
			self.values   += [self.A[x,y]]

	def runInterpolation(self, k = None, columnwise = False, show_prediction = False, use_obs = True):
		""" This function runs interpolation predictions as a baseline model for spectrogram predictions.

		Args:
			k: gap size, if applicable
			columnwise: #TODO
			show_prediction: Whether or not to plot the prediction.

		"""
		#2D interpolation
		#observer coordinates

		
		#if self.prediction_type == "block" or self.prediction_type == "exact":
		if type(self.resp_idx) == range:
			print("resp_idx: " + str(self.resp_idx))
			self.resp_idx = list(self.resp_idx)
		
		#missing_ = 60
		assert self.interpolation_method in ["griddata-linear", "rbf", "griddata-cubic", "griddata-nearest"]

		if self.interpolation_method	 in ["griddata-linear", "griddata-cubic", "griddata-nearest"]:
			#print(self.interpolation_method)
			points_to_predict, values, point_lst = [], [], []
			printc("self.prediction_type" + self.prediction_type, 'fail')
			if self.prediction_type == "block" or self.prediction_type == "exact" or self.prediction_type == "freqs":

				#Training points
				resp_idx = self.resp_idx
				obs_idx  = self.obs_idx
				#print("pred type: " + str(self.prediction_type))
				
				total_zone_idx = resp_idx + obs_idx
			
				#Train zone
				for x in range(self.xTr.shape[0]):
					# resonse points : train
					for y in total_zone_idx:
						point_lst += [(x,y)]#list(zip(range(Train.shape[0]) , [missing_]*Train.shape[0]))
						values	  += [self.A[x,y]]
						
				#Test zone
				for x in range(self.xTr.shape[0], self.A.shape[0]):
					# test set
					for y in resp_idx:
						points_to_predict += [(x,y)]
						
					# test set observers
					if use_obs:
						"using observers"
						for y in obs_idx:
							point_lst += [(x,y)]
							values	+= [self.A[x,y]]
					else:
						"not using observers"

			elif self.prediction_type == "column":
				

				total_zone_idx = range(self.A.shape[1])

				for x in range(self.xTr.shape[0]):

					# resonse points : train
					for y in total_zone_idx:
						
						point_lst += [(x,y)]
						values	  += [self.A[x,y]]
				
				for x in range(self.xTr.shape[0], self.xTr.shape[0] + self.xTe.shape[0]):
					# test set
					for y in total_zone_idx:
						points_to_predict += [(x,y)]

				xx_test, yy_test = list(zip(*points_to_predict)) 
				xx_train, yy_train = list(zip(*point_lst)) 

				if show_prediction:
					plt.xlim(0, max(xx_train+ xx_test))
					plt.ylim(0, max(yy_train + yy_test))
					plt.scatter(xx_train, yy_train, color = "blue")
					plt.scatter(xx_test, yy_test, color = "red")
					plt.show()	
				self.interpolation_method = "griddata-nearest"

			#extract the right method calls
			translation_dict = { "griddata-linear" : "linear", "griddata-cubic"  : "cubic", "griddata-nearest": "nearest"}	
			griddata_type = translation_dict[self.interpolation_method]
			if self.verbose:
				print("griddata " + griddata_type + " interpolation")
			
			ip2_pred = griddata(point_lst, values, points_to_predict, method = griddata_type)#"nearest")#griddata_type)#, rescale = True)#, method="linear")#"nearest")#"linear")#'cubic')
			
			#Shape(ip2_pred, "ip2_pred")
			#print(self.xTe.shape)
			ip2_pred = ip2_pred.reshape(self.xTe.shape)
			#ip2_resid = ip2_pred - self.xTe
			#points we can see in the training set

			if show_prediction:
				plt.imshow(ip2_pred, aspect = 0.1)
				plt.show()
				
				plt.imshow(self.xTe, aspect = 0.1)
				plt.show() 

			###plots:
			self.ip_res = {"prediction": ip2_pred, 
						   "nrmse" : nrmse(pred_ = ip2_pred, truth = self.xTe),
						   "method": translation_dict[self.interpolation_method],
						   "columnwise" : columnwise} 

		elif self.interpolation_method == "rbf":
			print("STARTING INTERPOLATION")
			total_zone_idx = range(self.A.shape[1])

			self.xs_known, self.ys_known, self.values, self.xs_unknown, self.ys_unknown  = [], [], [], [], []

			if self.prediction_type == "block":
				for x in range(self.xTr.shape[0]):
					# resonse points : train
					for y in total_zone_idx:
						self.rbf_add_point((x, y))

				#Test zone
				for x in range(self.xTr.shape[0], self.A.shape[0]):
					for y in resp_idx:		# test set
						self.rbf_add_point((x,y), test_set = True)
						
					for y in obs_idx:		# test set observers
						self.rbf_add_point((x,y))
			elif self.prediction_type == "column":
				"""
				for x in range(self.xTr.shape[0]):
						# resonse points : train
						for y in total_zone_idx:
							
							point_lst += [(x,y)]#list(zip(range(Train.shape[0]) , [missing_]*Train.shape[0]))
							values	  += [self.A[x,y]]
				
				#Test set doesn't depend on k
				for x in range(self.xTr.shape[0], self.xTr.shape[0] + self.xTe.shape[0]):
					# test set
					for y in total_zone_idx:
						points_to_predict += [(x,y)]
				"""
				for x in range(self.xTr.shape[0]):
					# resonse points : train
					for y in total_zone_idx:
						self.rbf_add_point((x, y))

				for x in range(self.xTr.shape[0], self.xTr.shape[0] + self.xTe.shape[0]):
					# test set
					for y in total_zone_idx:
						self.rbf_add_point((x, y))


			x, y, values = [np.array(i) for i in [self.xs_known, self.ys_known, self.values]]
			

			ALL_POINTS_AT_ONCE = False

			self.rbfi = Rbf(x, y, values, function='cubic') 

			print("RBF SET")

			if ALL_POINTS_AT_ONCE:

				xi, yi = [np.array(i) for i in [self.xs_unknown, self.ys_unknown]]

				#epsilon=2, #function = "cubic")  # radial basis function interpolato
				di   = rbfi(xi, yi) 				# interpolated values
				di   = di.reshape(self.xTe.shape)
				diR  = nrmse(pred_ = di, truth = self.xTe, columnwise = columnwise)

				self.ip_res = {"prediction" : di, "nrmse" : diR} 

				print("FINISHED INTERPOLATION: R = " + str(diR))

			else:
				values_rbf_output = []
				LEN = len(self.xs_unknown)
				for i, xi in enumerate(self.xs_unknown):
					print( i / LEN * 100)
					yi = np.array(self.ys_unknown[i])
					xi = np.array(xi)
					di = self.rbfi(xi, yi) 

					values_rbf_output +=[di]#[0]
				values_rbf_output = np.array(values_rbf_output).reshape(self.xTe.shape)
				diR  = nrmse(pred_ = di, truth = self.xTe, columnwise = columnwise)

				self.ip_res = {"prediction" : di, 
									"nrmse" : diR} 
				print("FINISHED INTERPOLATION: R = " + str(diR))
""" Hybrid: vestigal
		if self.model == "hybrid":
			print("old input dict: ")
			print(input_dict)
			old_bounds, old_bounds_keys  = cv_args["bounds"], list(cv_args["bounds"].keys())
			
			new_bounds = {}
			not_log_adjusted = ["n_nodes", "spectral_radius"]
			for i in old_bounds_keys:
				if i not in not_log_adjusted:
					print("adjusting " + i)
					new_bounds[i] = float(np.log(self.best_arguments[i])/np.log(10))
				else:
					new_bounds[i] = self.best_arguments[i]

			new_bounds['llambda'] = hybrid_llambda_bounds
			

			cv_args["bounds"] = new_bounds

			#print("HYBRID, New Bounds: " + str(cv_args["bounds"]))
			#print(cv_args)


			self.exp = True
			self.esn_cv.exp_weights = True
			exp_w_ = {'exp_weights' : True}

			#self.esn_cv = self.esn_cv_spec(**input_dict)
			if self.prediction_type == "column":
				self.best_arguments =  self.esn_cv.optimize(x = None, y = self.xTr) 
			else:
				self.best_arguments =  self.esn_cv.optimize(x = self.Train, y = self.xTr) 
		"""