from tqdm.notebook import trange, tqdm
import pickle
import pandas as pd
from PyFiles.experiment import *
from collections import defaultdict


def check_for_duplicates(lst, UnqLst = True, verbose = True):
    """ return the duplicate items in a list
    Args:
        UnqLst: if True return the duplicates
        verbose: if True print the duplicates
    """
    lst_tmp = []
    duplicates = []
    for i in lst:
        if i in lst_tmp:   
            duplicates += [i]
        else:
            lst_tmp += [i]
    if verbose == True:
        print(duplicates)
    if UnqLst:
        return(lst_tmp)

def build_string(message, *values, sep = ""): 
    """
    example_usage: build_string("bp_", 5, 6, 7, "blah", sep = "/")
    ARGUMENTS:
        message:
        values:
        sep:

    """
    if not values:
        return message
    else:
        return message.join(str(x) + sep for x in values)

class EchoStateAnalysis:
    """ #Spectrogram analysis class for analyzing the output of neural networks.
    
    Args:
        bounds:
        size: a string in ["small", "medium", "publish"] that refer to different dataset sizes.
        file_path: a string that describes the directory where the data is located. (load from)
        out_path: where to save the data
        target_frequency: in Hz which frequency we want to target as the center of a block experiment or the only frequency in the case of a simple prediction.
        bp: the base path
        force:
        path_lst:
        bp:
        data_type:
        model:


    """
    def __init__(self, path_list, ip_method = "linear", bp = None, 
                 force = False, ip_use_observers = True, data_type = "pickle", model = None, force_random_expers = False, over_ride_idx = None,
                 over_ride_best_args = None):
        self.path_list = path_list
        self.bp = bp
        self.force = force
        self.ip_use_observers = ip_use_observers
        self.ip_method = ip_method
        self.model = model
        self.data_type = data_type
        self.over_ride_idx = over_ride_idx
        self.over_ride_best_args = over_ride_best_args

        self.force_random_expers = force_random_expers

        assert model in ["uniform", "exponential","random", "delay_line", "cyclic"]

        print('path_list ', path_list)
        self.recover_old_data = False if 'exper_result' in self.load_data(path_list[0], bp = self.bp, verbose = False) else True

        self.change_interpolation_pickle()

        models = ["uniform", "exponential", "ip: linear"]

        if self.model in ["delay_line", "cyclic"]:
            models = [self.model, "ip: linear"]

        #self.build_loss_df(models = models, group_by = "freq", columnwise = False)
        #self.build_loss_df(models = models, group_by = "time", columnwise = False)
    def get_experiment(self, experiment_object, model, compare_ = False, plot_split = False, librosa = False, verbose = False):
        """Robust to change in data structure:
        
        self.get_experiment_new( exper_result = experiment_object, 
                                     model = model, 
                                     compare_ = compare_, 
                                     plot_split = plot_split, 
                                     verbose = verbose)
        """
        if not self.recover_old_data:
            print("GETTING NEW EXPERIMENT")
            self.get_experiment_new( exper_result = experiment_object, 
                                     model = model, 
                                     compare_ = compare_, 
                                     plot_split = plot_split, 
                                     verbose = verbose)
        else:
            
            experiment_ = self.get_experiment_old(pickle_obj=experiment_object,
                                    model=model, 
                                    compare_ = compare_, 
                                    plot_split = plot_split, 
                                    librosa = librosa, 
                                    verbose = verbose)

            return(experiment_)

    def get_experiment_new(self, exper_result, model, compare_ = False, plot_split = False, verbose = False, interpolation = False):
        """ 

        This function retrieves, from a json dictionary file, an EchoStateExperiment object.

        Args:
            json_obj: This variable name doesn't make sense. Change it.
            model: The model type. This needs to be adjusted.
            librosa:
            compare_: if True run the compare function above (plot the NRMSE comparison)

            plot_split: if True plot the 2D train-test split

            librosa: If True, load a pickle file (instead of json), perhaps other things specific to 
                the spectrograms that were created with the librosa package.

        """
        #https://towardsdatascience.com/design-optimization-with-ax-in-python-957b1fec776f

        model_result = exper_result.get_model_result(model)

        exper_inputs = model_result.EchoStateExperiment_inputs
        obs_inputs = exper_result.get_observers_input

        #if not interpolation:
        best_hyper_parameters = model_result.get_params()

        experiment_ = EchoStateExperiment( **exper_inputs )

        print('prediction_type', exper_inputs.prediction_type)

        obs_inputs['plot_split'] = plot_split

        printc("Reaching stage 2", 'fail')
        
        experiment_.get_observers(**obs_inputs)
        experiment_.already_trained( best_args = best_hyper_parameters, model = model_result)#self.model)


        Data = model_result.Data

        xx = range(Data.xTe.shape[0])

        if compare_:
            compare_inputs = {}

            #this is a hacky solution.
            changing_terminology = {"uniform" : "unif_w_pred", "exponential" : "exp_w_pred"}

            for model in list(json_obj["prediction"].keys()):
                specific_prediction = json_obj[model]
                specific_key = changing_terminology[model]
                add_modelcompare_inputs[specific_key] = json_obj["prediction"][model]


            unif_w_pred, exp_w_pred = json_obj["prediction"]["uniform"], json_obj["prediction"]["exponential"]
            ip_pred = json_obj["prediction"]["interpolation"]
            unif_w_pred, exp_w_pred, ip_pred = [np.array(i) for i in [unif_w_pred, exp_w_pred, ip_pred]]

            compare( truth = np.array(experiment_.Test), unif_w_pred = unif_w_pred, ip_pred = ip_pred,
                exp_w_pred  = exp_w_pred, columnwise  = False, verbose = False)
            if n_keys == 2:
                compare(
                    truth       = np.array(experiment_.Test), 
                    unif_w_pred = np.array(json_obj["prediction"]["uniform"]),
                    ip_pred = np.array(json_obj["prediction"]["interpolation"]),
                    exp_w_pred  = None,#np.array(json_obj["prediction"]["exponential"]), 
                    columnwise  = False,
                    verbose = False)
        if verbose == True:
            """
            for i in [[unif_w_pred, "unif pred"],
                      [exp_w_pred, "exp pred"],
                      [ip_pred, "ip pred"],
                      [np.array(experiment_.Test), "Test" ]]:
                Shape(i)
            """
            print("experiment inputs: " + str(json_obj["experiment_inputs"]))
            print("get_obs_inputs: " + str(obs_inputs))
            print("Train.shape: " + str(experiment_.Train.shape))
            print("Saved_prediction.shape: " + str(np.array(json_obj["prediction"]["uniform"]).shape))
        return(experiment_)


    def get_experiment_old(self, pickle_obj, model, compare_ = False, plot_split = False, librosa = False, verbose = False):
        """ 

        This function retrieves, from a json dictionary file, an EchoStateExperiment object.

        Args:
            json_obj: This variable name doesn't make sense. Change it.
            model: The model type. This needs to be adjusted.
            librosa:
            compare_: if True run the compare function above (plot the NRMSE comparison)

            plot_split: if True plot the 2D train-test split

            librosa: If True, load a pickle file (instead of json), perhaps other things specific to 
                the spectrograms that were created with the librosa package.

        """
        #https://towardsdatascience.com/design-optimization-with-ax-in-python-957b1fec776f

        print("get_experiment_old model: ", model)
        if self.over_ride_idx:
            pickle_obj["experiment_inputs"]["obs_idx"], pickle_obj["experiment_inputs"]["resp_idx"] = self.over_ride_idx


        if model in ["exponential", "uniform"]:
            obs_inputs = pickle_obj["get_observer_inputs"]
            vals = pickle_obj["best arguments"][model]

            if self.over_ride_best_args:
                if model in self.over_ride_best_args.keys():
                    vals = self.over_ride_best_args[model]
            print("best_args", vals)
            extra_inputs = { 
                         "prediction_type" : "exact",
                         "model" : "random",
                         "input_weight_type" : model
                    }
            

        if obs_inputs["method"] != "column":
            obs_inputs["method"] = "exact"


       

        #printc("experiment inputs" +str(json_obj["experiment_inputs"]), 'fail')

        printc("experiment inputs" +str(pickle_obj["get_observer_inputs"]), 'fail')

        

        experiment_ = EchoStateExperiment(**pickle_obj["experiment_inputs"], **extra_inputs,  librosa = librosa)
        
        experiment_.get_observers(**obs_inputs, plot_split = plot_split)
        experiment_.already_trained(vals, model = model)



        #def already_trained(self, best_args, model):

        xx = range(experiment_.xTe.shape[0])

        if compare_:
            compare_inputs = {}

            #this is a hacky solution.
            changing_terminology = {"uniform" : "unif_w_pred", "exponential" : "exp_w_pred"}

            for model in list(json_obj["prediction"].keys()):
                specific_prediction = json_obj[model]
                specific_key = changing_terminology[model]
                compare_inputs[specific_key] = json_obj["prediction"][model]


            unif_w_pred, exp_w_pred = json_obj["prediction"]["uniform"], json_obj["prediction"]["exponential"]
            ip_pred = json_obj["prediction"]["interpolation"]
            unif_w_pred, exp_w_pred, ip_pred = [np.array(i) for i in [unif_w_pred, exp_w_pred, ip_pred]]

            compare( truth = np.array(experiment_.Test), unif_w_pred = unif_w_pred, ip_pred = ip_pred,
                exp_w_pred  = exp_w_pred, columnwise  = False, verbose = False)
            if n_keys == 2:
                compare(
                    truth       = np.array(experiment_.Test), 
                    unif_w_pred = np.array(json_obj["prediction"]["uniform"]),
                    ip_pred = np.array(json_obj["prediction"]["interpolation"]),
                    exp_w_pred  = None,#np.array(json_obj["prediction"]["exponential"]), 
                    columnwise  = False,
                    verbose = False)
        if verbose == True:
            """
            for i in [[unif_w_pred, "unif pred"],
                      [exp_w_pred, "exp pred"],
                      [ip_pred, "ip pred"],
                      [np.array(experiment_.Test), "Test" ]]:
                Shape(i)
            """
            print("experiment inputs: " + str(json_obj["experiment_inputs"]))
            print("get_obs_inputs: " + str(obs_inputs))
            print("Train.shape: " + str(experiment_.Train.shape))
            print("Saved_prediction.shape: " + str(np.array(json_obj["prediction"]["uniform"]).shape))

        return(experiment_)
    

    def change_interpolation_pickle(self):
        """
        #TODO impliment robust version. or impliment a function that can recover the old data.
        """
        if not self.recover_old_data:
            self.change_interpolation_pickle_new()
        else:
            self.change_interpolation_pickle_old()

    def change_interpolation_pickle_new(self, verbose = False):
        """
        Force_random__expers: added 11/21, this will not allow the model to build random experiments if they don't include exponential weights.

        """
        path_lst = self.path_list

        #strange, this seems hacky.
        if self.ip_use_observers == False:
            self.ip_method = "nearest"

        #What is the point of this loop? If model is random -->  make sure uniform and exponential are successfully run.
        #also a vestigal part of this loop is to check if it is even possible to load this.
        
        #print()

        self.experiment_results = []


        for i, _ in enumerate(trange(len(path_lst), desc='experiment list, loading data...')): 
            
            experiment_result = self.load_data(path_lst[i], bp = self.bp, verbose = False)

            self.experiment_results.append(experiment_result["exper_result"])

        printc('experiment_results ' + str(self.experiment_results), 'fail')

            
        for i, _ in enumerate(trange( len(self.experiment_results), desc='experiment list, fixing interpolation...')): 

            try:
                #live_nns = self.fix_Rcs(experiment_results[i])
                printc("Let's not fix something that ain't broken.", 'green')
                break
            except:

                #This code, aside from the barplots, is essentially trying to solve a problem via repair rather than 
                #fixing the problem at the source. Let's instead fix the problem at the source.

                experiment_dict, experiment_obj = self.fix_Rcs(experiment_results[i])
                printc("failing to retrieve live nns, ... recalibrating to old data structure", 'red')

                if self.ip_method == "all":
                    ip_methods_ = ["linear", "cubic", "nearest"]
                else:
                    ip_methods_ = [self.ip_method]

                for ip_method_ in ip_methods_:
                    tmp_dict = self.fix_interpolation(experiment_dict, experiment_obj, method = ip_method_)
                    experiment_dict["nrmse"]["ip: " + ip_method_] = tmp_dict["nrmse"]["interpolation"]
                    experiment_dict["prediction"]["ip: " + ip_method_] = tmp_dict["prediction"]["interpolation"]
                try:
                    rel_spec = { key: experiment_dict["nrmse"][key] / experiment_dict["nrmse"]["ip: linear"] 
                                for key in experiment_dict["nrmse"].keys()}
                except:
                    rel_spec = { key: experiment_dict["nrmse"][key] / experiment_dict["nrmse"]["interpolation"] 
                                for key in experiment_dict["nrmse"].keys()}
                
                results_rel.append(rel_spec)
                
                #removing interpolation
                for key in ["prediction", "nrmse"]:
                    dict_tmp = experiment_dict[key]
                    dict_tmp = ifdel(dict_tmp, "interpolation")
                    experiment_dict[key] = dict_tmp


                results_tmp.append(experiment_dict["nrmse"])
                experiment_list.append(experiment_dict)

                results_df = pd.DataFrame(results_tmp)
                results_df = results_df.melt()
                results_df.columns = ["model", "R"]

                results_df_rel = pd.DataFrame(results_rel)
                results_df_rel = results_df_rel.melt()
                results_df_rel.columns = ["model", "R"]
                
                self.experiment_lst = experiment_list
                self.R_results_df = results_df
                self.R_results_df_rel = results_df_rel

                if NOT_YET_RUN:
                    print("the following paths have not yet been run: ")
                    print(np.array(path_lst)[NOT_YET_RUN])
                   
                        
                if NOT_INCLUDED:
                    print("the following paths contain incomplete experiments: (only unif finished)")
                    #print(np.array(path_lst_unq)[NOT_INCLUDED])
                    print(np.array(path_lst)[NOT_INCLUDED])
                    

                NOT_INCLUDED = check_for_duplicates(NOT_INCLUDED)
                NOT_YET_RUN = check_for_duplicates(NOT_YET_RUN)
                print("total experiments completed: " + str(len(self.experiment_lst)))
                print("total experiments half complete: " + str(len(NOT_INCLUDED)))
                print("total experiments not yet run: " + str(len(NOT_YET_RUN)))
                pct_complete = (len(self.experiment_lst))/(len(self.experiment_lst)+len(NOT_INCLUDED)*0.5 + len(NOT_YET_RUN)) * 100
                pct_complete  = str(round(pct_complete, 1))
                print("Percentage of tests completed: " + str(pct_complete) + "%")  

    def change_interpolation_pickle_old(self, verbose = False):
        """
        Force_random__expers: added 11/21, this will not allow the model to build random experiments if they don't include exponential weights.

        """
        path_lst = self.path_list

        #strange, this seems hacky.
        if self.ip_use_observers == False:
            self.ip_method = "nearest"

        #What is the point of this loop? If model is random -->  make sure uniform and exponential are successfully run.
        #also a vestigal part of this loop is to check if it is even possible to load this.
        
        

        for i, _ in enumerate(trange(len(path_lst), desc='experiment list, loading data...')): 
            

            if i == 0:
                results_tmp, results_rel = [], []
                experiment_list_temp, NOT_INCLUDED, NOT_YET_RUN  = [], [], []

            experiment_dict = self.load_data(path_lst[i], bp = self.bp, verbose = False)
            print('experiment_dict:', experiment_dict)
            if verbose:
                print(list(experiment_dict["prediction"].keys()))
            if self.model in ["delay_line", "cyclic"] or self.force_random_expers:
                experiment_list_temp.append(experiment_dict)
            else:
                try:
                    models_spec = list(experiment_dict["prediction"].keys())
                    assert "exponential" in models_spec, print(models_spec)
                    try:
                        assert len(models_spec) >= 3
                        experiment_list_temp.append(experiment_dict)
                    except:
                        if verbose:
                            print("NOT INCLUDED")
                        NOT_INCLUDED += [i]
                except:
                    if verbose:
                        print("NOT YET RUN")
                    NOT_YET_RUN += [i]

        for i, _ in enumerate(trange( len(experiment_list_temp), desc='experiment list, fixing interpolation...')): 

            if not i:
                results_tmp, results_rel, experiment_list = [], [], []

            experiment_dict = experiment_list_temp[i]

            printc("experiment_dict " + str(experiment_dict), 'green')

            experiment_dict, experiment_obj = self.fix_Rcs(experiment_dict)
            if self.ip_method == "all":
                ip_methods_ = ["linear", "cubic", "nearest"]
            else:
                ip_methods_ = [self.ip_method]

            for ip_method_ in ip_methods_:
                tmp_dict = self.fix_interpolation(experiment_dict, experiment_obj, method = ip_method_)
                experiment_dict["nrmse"]["ip: " + ip_method_] = tmp_dict["nrmse"]["interpolation"]
                experiment_dict["prediction"]["ip: " + ip_method_] = tmp_dict["prediction"]["interpolation"]
            try:
                rel_spec = { key: experiment_dict["nrmse"][key] / experiment_dict["nrmse"]["ip: linear"] 
                            for key in experiment_dict["nrmse"].keys()}
            except:
                rel_spec = { key: experiment_dict["nrmse"][key] / experiment_dict["nrmse"]["interpolation"] 
                            for key in experiment_dict["nrmse"].keys()}
            
            results_rel.append(rel_spec)
            
            #removing interpolation
            for key in ["prediction", "nrmse"]:
                dict_tmp = experiment_dict[key]
                dict_tmp = ifdel(dict_tmp, "interpolation")
                experiment_dict[key] = dict_tmp


            results_tmp.append(experiment_dict["nrmse"])
            experiment_list.append(experiment_dict)

        results_df = pd.DataFrame(results_tmp)
        results_df = results_df.melt()
        results_df.columns = ["model", "R"]

        results_df_rel = pd.DataFrame(results_rel)
        results_df_rel = results_df_rel.melt()
        results_df_rel.columns = ["model", "R"]
        
        self.experiment_lst = experiment_list
        self.R_results_df = results_df
        self.R_results_df_rel = results_df_rel

        if NOT_YET_RUN:
            print("the following paths have not yet been run: ")
            print(np.array(path_lst)[NOT_YET_RUN])
           
                
        if NOT_INCLUDED:
            print("the following paths contain incomplete experiments: (only unif finished)")
            #print(np.array(path_lst_unq)[NOT_INCLUDED])
            print(np.array(path_lst)[NOT_INCLUDED])
            

        NOT_INCLUDED = check_for_duplicates(NOT_INCLUDED)
        NOT_YET_RUN = check_for_duplicates(NOT_YET_RUN)
        print("total experiments completed: " + str(len(self.experiment_lst)))
        print("total experiments half complete: " + str(len(NOT_INCLUDED)))
        print("total experiments not yet run: " + str(len(NOT_YET_RUN)))
        pct_complete = (len(self.experiment_lst))/(len(self.experiment_lst)+len(NOT_INCLUDED)*0.5 + len(NOT_YET_RUN)) * 100
        pct_complete  = str(round(pct_complete, 1))
        print("Percentage of tests completed: " + str(pct_complete) + "%")  
    
    def load_data(self, file = "default", print_lst = None, bp = None, verbose = False, enforce_exp = False):
        """
        print_lst can contain a list of keys to print, for example ["nrmse"]

        Args:
            file:
            print_lst:
            bp:
            verbose:
            enforce_exp:
        """

        if bp != None:
            file = bp + file
        if file == "default":
            nf = get_new_filename(exp = exp, current = True)
        else:
            nf = file
        if ".pickle" in file:
            with open(nf, 'rb') as f:
                datt = pickle.load(f)
        else:
            with open(nf) as json_file: # 'non_exp_w.txt'
                datt = json.load(json_file)
            print(datt["nrmse"])
            
        if verbose:
            print(datt["nrmse"])
        
        return(datt)
    
    def compare(self, truth, unif_w_pred = None, exp_w_pred = None, ip_pred = None, 
                columnwise = False, verbose = False):
        """ This function compares the NRMSE of the various models: RC [exp, unif] and interpolation

        Args:

            columnwise: This function provides two things, conditional on the columnwise variable.
                False: cross-model comparison of nrmse
                True: model nrmse correlary for each point.

            ip_pred: The interpolation prediction over the test set
            exp_w_pred: Exponential weight RC prediction
            unif_w_pred: Uniform weight RC prediction
        """
        nrmse_inputs = {"truth" : truth, "columnwise" : columnwise}
        pred_list = [unif_w_pred, exp_w_pred, ip_pred]
        pred_dicts = [{"pred_": prediction, **nrmse_inputs} for prediction in pred_list]

        def conditional_nrmse(inputs):
            """ If the prediction exists, calculate the NRMSE

            Args: 
                inputs: a dictionary {truth, pred_}
            """
            return None if not inputs["pred_"] else nrmse(**inputs) # check if prediction is empty


        unif_nrmse, exp_nrmse, ip_nrmse = [nrmse(**i) for i in pred_dicts]


        hi = """
        if type(unif_w_pred) != type(None):
            unif_nrmse = nrmse(pred_ = unif_w_pred, **nrmse_inputs)

        if type(exp_w_pred) != type(None):
            exp_nrmse  = nrmse(pred_  = exp_w_pred , **nrmse_inputs)

        if type(ip_pred) != type(None):
            ip_nrmse   = nrmse(pred_  = ip_pred , **nrmse_inputs)
        """
        ip_res = {"nrmse" : ip_nrmse, "pred" : ip_pred}


        assert type(columnwise) == bool, "columnwise must be a boolean"

        if not columnwise:
            if verbose:
                print("cubic spline interpolation nrmse: " + str(ip_res["nrmse"]))
                print("uniform weights rc nrmse: " + str(unif_nrmse))
                print("exponential weights rc nrmse: " + str(exp_nrmse))
                print("creating barplot")

            unif_and_ip_dict = {"interpolation" : ip_res["nrmse"], "uniform rc" : unif_nrmse}
            exp_dict = {"exponential rc" : exp_nrmse}

            if type(exp_w_pred) != type(None):
                df = pd.DataFrame({"interpolation" : ip_res["nrmse"], 
                                   "uniform rc" : unif_nrmse, 
                                   "exponential rc" : exp_nrmse}, index = [0])
            else:
                df = pd.DataFrame({"interpolation" : ip_res["nrmse"], 
                                   "uniform rc" : unif_nrmse}, index = [0])
            display(df)

            plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            sns.catplot(data = df, kind = "bar")
            plt.title("model vs nrmse")
            plt.ylabel("nrmse")
            improvement = []
            for rc_nrmse in[unif_nrmse, exp_nrmse]:
                impr_spec = ((ip_res["nrmse"] - rc_nrmse)/ip_res["nrmse"]) * 100
                impr_spec = [round(impr_spec,1)]
                improvement += impr_spec

            pct_improve_unif, pct_improve_exp = improvement
            if pct_improve_unif > 0:
                print("unif improvement vs interpolation: nrmse " + str(-pct_improve_unif) + "%")
            else:
                print("rc didn't beat interpolation: nrmse +" + str(-pct_improve_unif) + "%")

            if pct_improve_exp > 0:
                print("exp improvement vs interpolation: nrmse " + str(-pct_improve_exp) + "%")
            else:
                print("rc didn't beat interpolation: nrmse +" + str(-pct_improve_exp) + "%")

            impr_rc_compare = round(((unif_nrmse - exp_nrmse)/unif_nrmse) * 100,1)

            if impr_rc_compare > 0:
                print("exp rc improvement vs unif rc: nrmse " + str(-impr_rc_compare) + "%")
            else:
                print("exp weights didn't improve rc: nrmse +" + str(-impr_rc_compare) + "%")
        else:
            print("creating first figure")
            model_names = ["interpolation", "uniform rc", "exponential rc"]
            for i, model_rmse_np in enumerate([ip_res["nrmse"], unif_nrmse, exp_nrmse]):
                model_rmse_pd = pd.melt(pd.DataFrame(model_rmse_np.T))
                model_rmse_pd.columns = ["t","y"]
                model_rmse_pd["model"] = model_names[i]
                if not i: # check if i == 0
                    models_pd = model_rmse_pd
                else:
                    models_pd = pd.concat([models_pd, model_rmse_pd ], axis = 0)
            fig, ax = plt.subplots(1,1, figsize = (11, 6))
            sns.lineplot(x = "t", y = "y", hue = "model", data = models_pd, ax = ax)
            ax.set_title("model vs rmse")
            ax.set_ylabel("nrmse")
            ax.set_xlabel("Test idx")

    def load_p_result (path : str, bp = ""):
        """ Load a pickle spectrogram result.

        Args:
            path: the path to the file
            bp: base_path
        """
        path = bp + path
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        return(b)

    def fix_Rcs(self, exper_):
       """
       this properly fixes the RC predictions after the output from experiment.py. FIX THIS UPSTREAM DAMNIT

       Args:
           exper_: the experiment dict being fed in.
       """
       for model in list(exper_['best arguments'].keys()):
           
           exper_spec = self.get_experiment(exper_, model, verbose = False, plot_split = False, compare_ = False)

           if "prediction" not in exper_.keys():
               exper_['prediction'] = {}
           exper_["prediction"][model] = exper_spec.prediction
           exper_["nrmse"][model] = nrmse(exper_spec.prediction, exper_spec.xTe)

       if self.over_ride_idx:
           exper_["obs_idx"], exper_["resp_idx"] = self.over_ride_idx
       #get important pieces of information for later, to avoid running get_experiment over and over and over.
       exper_["xTe"] = exper_spec.xTe
       exper_["xTr"] = exper_spec.xTr
       exper_["f"]   = exper_spec.f
       exper_["T"]   = exper_spec.T
       exper_["A"]   = exper_spec.A

       exper_dict = exper_
       exper_obj = exper_spec

       return(exper_dict, exper_spec)

    """
    def fix_Rcs(self, exper_):
        
        if not self.recover_old_data:
            exper_ = exper_["exper_result"]
            live_nns = []
            for model, _ in exper_.model_results.items():
                exper_spec = self.get_experiment(exper_, model, verbose = False, plot_split = False, compare_ = False)
                live_nns.append(live_nns)
            return(live_nns)

        #old way: 12/2/2020
        else:
            for model in list(exper_['best arguments'].keys()):
                exper_spec = self.get_experiment(exper_, model, verbose = False, plot_split = False, compare_ = False)

                if not exper_["prediction"]:
                    exper_["prediction"] = {}
                if not exper_["nrmse"]:
                    exper_["nrmse"] = {}
                exper_["prediction"][model] = exper_spec.prediction
                exper_["nrmse"][model] = nrmse(exper_spec.prediction, exper_spec.xTe)

            #get important pieces of information for later, to avoid running get_experiment over and over and over.
            exper_["xTe"] = exper_spec.xTe
            exper_["xTr"] = exper_spec.xTr
            exper_["f"]   = exper_spec.f
            exper_["T"]   = exper_spec.T
            exper_["A"]   = exper_spec.A

            exper_dict = exper_
            exper_obj = exper_spec

            return(exper_dict, exper_spec)
    """


    def fix_interpolation(self, exper_, exper_spec, method):
        """ Change the interpolation method of the inputed experiment.

        Args:
            exper_: the json experiment dictionary which you will to alter.
            exper_spec: the echoStateExperiment object from which to run the interpolation method.
            method: the type of interpolation method (chosen from 'linear', 'cubic', 'nearest')
        """
        # we can get most of our information from either model, we just have to fix the exponential predictions at the end.

        #interpolation:
        if method == "cubic":
            exper_spec.interpolation_method = "griddata-cubic"
        elif method == "nearest":
            exper_spec.interpolation_method = "griddata-nearest"
        exper_spec.runInterpolation(use_obs = self.ip_use_observers)
        exper_["prediction"]["interpolation"] = exper_spec.ip_res["prediction"]
        exper_["nrmse"]["interpolation"] = exper_spec.ip_res["nrmse"]

        #now repair the exponential predictions.
        return(exper_)
        

    
    def make_R_barplots(self, label_rotation = 45):
        """
        Self.Parameters:
            self.R_results_df is the nrmse pd dataframe, non-columnwise for all experiments in path_lst
            self.R_resulsts_df_rel is the relative R pd dataframe
        """
        fig, ax = plt.subplots(1, 2, figsize = (14, 5))
        ax = ax.ravel()

        sns.violinplot(x = "model", y = "R", data = self.R_results_df, ax=ax[0])
        sns.violinplot(x = "model", y = "R", data = self.R_results_df_rel, ax=ax[1])
        for i in range(len(ax)):
            ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=label_rotation)
        ax[0].set_ylim(0, 1.05)
        ax[1].set_ylim(0.5, 2.0)

    def loss(self, pred_, truth, typee = "L1"):
        """
        inputs should be numpy arrays
        variables:
        pred_ : the prediction numpy matrix
        truth : ground truth numpy matrix
        columnwise: bool if set to true takes row-wise numpy array (assumes reader thinks of time as running left to right
            while the code actually runs vertically.)
            
        This is an average of the loss across that point, which we must do if we are to compare different sizes of data.

        """
        pred_ = np.array(pred_)
        truth = np.array(truth)
        assert pred_.shape == truth.shape, "pred shape: " + str(pred_.shape) + " , " + str(truth.shape)

        def L2_(pred, truth):
            resid = pred - truth
            return(resid**2)

        def L1_(pred, truth):
            resid = pred - truth
            return(abs(resid))

        def R_(pred, truth):
            loss = ((pred - truth)**2)/(np.sum(truth**2))
            loss = np.sqrt(loss)
            return(loss)
            
        assert typee in ["L1", "L2", "R"]
        if typee == "L1":
            f = L1_
        elif typee == "L2":
            f = L2_
        elif typee == "R":
            f = R_
            
        loss_arr = f(pred_, truth )  

        return(loss_arr)

    

    def build_loss_df_new(self, group_by = "time", columnwise = True, relative = True,
                  rolling = None, models = ["uniform", "exponential", "interpolation"],
                  silent = True, hybrid = False):
        #exp stands for experiment here, not exponential
        """ Builds a loss dataframe
        
        Args:
            #TODO
        columnwise == False means don't take the mean.
        """

        def flatten_loss_dict(loss_dict, model, time_lst, freq_lst, experiment_num = 0):
            """

            """
            new_dict = {"freq" : [],
                        "time" : [],
                        "experiment #" : experiment_num,
                        "model" : model}
            
            for key in loss_dict.keys():
                new_dict[key] = []

            for j, (loss_key, loss_Arr) in enumerate(loss_dict.items()):
                for i in range(loss_Arr.shape[1]):
                    spec_ = loss_Arr[:,i].reshape(-1,)
                    if not j:
                        new_dict["freq"] += list(np.full(spec_.shape, fill_value = freq_lst[i]))
                        new_dict["time"] += list(time_lst)
                    new_dict[loss_key] += list(spec_)
            pd_ = pd.DataFrame(new_dict)
            return(pd_)

        experiment_list = self.experiment_results
        
        count = 0
        for i in trange(len(experiment_list), desc='processing path list...'):

            experiment_ = experiment_list[i]

            if i == 0:


                get_observers_input = experiment_.get_observers_input
                train_, test_ = experiment_.data.Target_Tr_, experiment_.data.Target_Te_
                resp_idx = experiment_.data.sets["Target_Te"].y_indices_
                time_idx = experiment_.data.sets["Target_Te"].time_indices_
                T, f = experiment_.data.T_, np.array(experiment_.data.f_)
                A = experiment_.data.A_
                existing_models = experiment_.get_models()


            if "split" in get_observers_input:
                split_ = get_observers_input["split"]
            
            #construct the required data frame and caculate the nrmse from the predictions:
                
            test_len = test_.shape[0]
            train_len = A.shape[0] - test_len
            
            ###################################################
            time_lst, freq_lst = [], [] 

            time_lst_one_run = list(T[time_idx].reshape(-1,))
            freq_lst_one_run = list(f[resp_idx].reshape(-1,))
            
            ###################################################

            
            

            for j, model in enumerate(existing_models):#enumerate(models):
                printc(model, 'green')
                try:
                    model_result_ = experiment_.get_model_result(model)
                    
                    shared_args = {
                        "pred_" : model_result_.prediction,
                        "truth": test_}
                except:

                    shared_args = {
                        "pred_" : experiment_["prediction"][model],
                        "truth": test_}

                self.L1_entire_df = self.loss(**shared_args, typee = "L1")
                self.L2_entire_df = self.loss(**shared_args, typee = "L2")
                self.R_entire_df  = self.loss(**shared_args, typee = "R")

                loss_arr_dict = {}

                for loss_metric in ["L1", "L2", "R"]:
                    loss_arr_dict[loss_metric] = self.loss(**shared_args, 
                                                           typee = loss_metric)
                
                rDF_spec = flatten_loss_dict(loss_arr_dict, 
                                              model = model,
                                              time_lst = list(np.linspace(time_lst_one_run[0],
                                                                          time_lst_one_run[-1],
                                                                          test_len)),
                                              freq_lst = freq_lst_one_run,  
                                              experiment_num = i)
                if count == 0:
                    self.rDF = rDF_spec
                else:
                    self.rDF = pd.concat([self.rDF, rDF_spec], axis = 0)
                count += 1 
        if silent != True:
            display(rDF)




    def build_loss_df_old(self, group_by = "time", columnwise = True, relative = True,
                  rolling = None, models = ["uniform", "exponential", "interpolation"],
                  silent = True, hybrid = False):
        #exp stands for experiment here, not exponential
        """ Builds a loss dataframe
        
        Args:
            #TODO
        columnwise == False means don't take the mean.
        """

        def flatten_loss_dict(loss_dict, model, time_lst, freq_lst, experiment_num = 0):
            """

            """
            new_dict = {"freq" : [],
                        "time" : [],
                        "experiment #" : experiment_num,
                        "model" : model}
            
            for key in loss_dict.keys():
                new_dict[key] = []

            for j, (loss_key, loss_Arr) in enumerate(loss_dict.items()):
                for i in range(loss_Arr.shape[1]):
                    spec_ = loss_Arr[:,i].reshape(-1,)
                    if not j:
                        new_dict["freq"] += list(np.full(spec_.shape, fill_value = freq_lst[i]))
                        new_dict["time"] += list(time_lst)
                    new_dict[loss_key] += list(spec_)
            pd_ = pd.DataFrame(new_dict)
            return(pd_)

        experiment_list = self.experiment_lst
        count = 0
        for i in trange(len(experiment_list), desc='processing path list...'):

            experiment_ = experiment_list[i]
            #exp_obj = self.get_experiment(exper_dict_spec, compare_ = False, verbose = False, plot_split = False)
            
            split_ = experiment_['get_observer_inputs']["split"]
            
            #construct the required data frame and caculate the nrmse from the predictions:
            train_, test_ = experiment_["xTr"], experiment_["xTe"]
           
                
            #if "hybrid" in exp_json["prediction"]:
            #    del exp_json["prediction"]['hybrid'] 
            #    del exp_json["nrmse"]['hybrid'] 

            if i == 0:
                A = experiment_["A"]
                
            test_len = test_.shape[0]
            train_len = A.shape[0] - test_len
            
            ###################################################
            time_lst, freq_lst = [], []
            resp_idx = experiment_["resp_idx"]
            T, f = experiment_["T"], np.array(experiment_["f"])

            time_lst_one_run = list(T[train_len:].reshape(-1,))
            freq_lst_one_run = list(f[resp_idx].reshape(-1,))
            ###################################################

            existing_models = experiment_["nrmse"].keys()
            

            for j, model in enumerate(models):

                
                shared_args = {
                    "pred_" : experiment_["prediction"][model],
                    "truth": test_}

                self.L1_entire_df = self.loss(**shared_args, typee = "L1")
                self.L2_entire_df = self.loss(**shared_args, typee = "L2")
                self.R_entire_df  = self.loss(**shared_args, typee = "R")

                loss_arr_dict = {}

                for loss_metric in ["L1", "L2", "R"]:
                    loss_arr_dict[loss_metric] = self.loss(**shared_args, 
                                                           typee = loss_metric)
                
                rDF_spec = flatten_loss_dict(loss_arr_dict, 
                                              model = model,
                                              time_lst = list(np.linspace(time_lst_one_run[0],
                                                                          time_lst_one_run[-1],
                                                                          test_len)),
                                              freq_lst = freq_lst_one_run,  
                                              experiment_num = i)
                if count == 0:
                    self.rDF = rDF_spec
                else:
                    self.rDF = pd.concat([self.rDF, rDF_spec], axis = 0)
                count += 1 
        if silent != True:
            display(rDF)
    
    def build_loss_df(self, group_by = "time", columnwise = True, relative = True,
                  rolling = None, models = ["uniform", "exponential", "interpolation"],
                  silent = True, hybrid = False):
        """try:
            build_loss_df_new(group_by = group_by, columnwise = columnwise , relative = relative,
                  rolling = rolling ,
                  silent = silent, hybrid = hybrid)
        except:
            build_loss_df_old(group_by = group_by, columnwise = columnwise , relative = relative,
                  rolling = rolling , models = ["uniform", "exponential", "interpolation"],
                  silent = silent, hybrid = hybrid)"""
        if not self.recover_old_data:

            self.build_loss_df_new(group_by = group_by, columnwise = columnwise , relative = relative,
                      rolling = rolling, silent = silent, hybrid = hybrid, models = models)
        else:
            #models.append("ip: linear")
            #models.remove("interpolation")
            self.build_loss_df_old(group_by = group_by, columnwise = columnwise , relative = relative,
                      rolling = rolling, silent = silent, hybrid = hybrid, models = models)

    def loss_plot(self, rolling, split, loss = "R",
                 relative = False, include_ip = True, hybrid = False, group_by = "freq"):

        rDF = self.rDF_time if group_by == "freq" else rDF_freq
        
        if not hybrid:
            rDF = rDF[rDF.model != "hybrid"]
        if include_ip == False:
            rDF = rDF[(rDF.model == "uniform") | (rDF.model == "exponential")]
        
        rDF = rDF[rDF.split == split]
        if loss == "L2":
            LOSS = rDF.L2_loss
        elif loss =="L1":
            LOSS = rDF.L1_loss
        elif loss =="R":
            LOSS = rDF.L1_loss
        
        fig, ax = plt.subplots(1, 1, figsize = (12, 6))
        if relative == True:
            diff = rDF[rDF.model == "exponential"]["loss"].values.reshape(-1,) - rDF[rDF.model == "uniform"]["loss"].values.reshape(-1,)

            df_diff = rDF[rDF.model == "uniform"].copy()
            df_diff.model = "diff"
            df_diff.loss = diff
            
            sns.lineplot( x = "time", y = "loss" , hue = "model" , data = df_diff)
            ax.set_title(loss_ + " loss vs time relative")
        else:
        
            display(rDF)
            if rolling != None:

                sns.lineplot( x = "time", y = LOSS.rolling(rolling).mean() , hue = "model" , data = rDF)
                #sns.scatterplot( x = "time", y = "loss" , hue = "model" , data = rDF, alpha = 0.005, edgecolor= None)
            else:
                sns.lineplot( x = "time", y = loss , hue = "model" , data = rDF)
            ax.set_title( "mean " + loss + " loss vs time for all RC's, split = " + str(split))
            ax.set_ylabel(loss)

    def hyper_parameter_plot(self):
        """
        Let's visualize the hyper-parameter plots.
        """
        log_vars = ["noise", "connectivity", "regularization", "llambda", "llamba2", "cyclic_res_w", "cyclic_input_w"]
        
        for i, experiment in enumerate(self.experiment_lst):
            df_spec_unif = pd.DataFrame(experiment["best arguments"]["uniform"], index = [0])
            df_spec_exp  = pd.DataFrame(experiment["best arguments"]["exponential"], index = [0])
            #if "cyclic" in list(experiment["best arguments"].keys()):
            df_spec_cyclic  = pd.DataFrame(experiment["best arguments"]["cyclic"], index = [0])
            
            if not i:
                df_unif = df_spec_unif
                df_exp = df_spec_exp
                df_cyclic = df_spec_cyclic
            else:
                df_unif = pd.concat([df_unif, df_spec_unif])
                df_exp = pd.concat([df_exp, df_spec_exp])
                df_cyclic = pd.concat([df_cyclic, df_spec_cyclic])

        unif_vars = ["connectivity", "regularization", "leaking_rate", "spectral_radius"]
        exp_vars  = ["llambda", "llambda2", "noise"]
        display(df_spec_cyclic)
        cyclic_vars = ["cyclic_res_w", "cyclic_input_w", "cyclic_bias", "leaking_rate"]
        #cyclic_vars  = list(experiment["best arguments"]["cyclic"].keys())
        df_unif = df_unif[unif_vars]
        df_exp = df_exp[unif_vars + exp_vars]
        df_cyclic = df_cyclic[cyclic_vars]

        
        for i in list(df_unif.columns):
            if i in log_vars:
                df_unif[i] = np.log(df_unif[i])/np.log(10)
                
        for i in list(df_exp.columns):
            if i in log_vars:
                df_exp[i] = np.log(df_exp[i])/np.log(10)

        for i in list(df_cyclic.columns):
            if i in log_vars:
                df_cyclic[i] = np.log(df_cyclic[i])/np.log(10)
        
        
        #display(df_unif)
        
        sns.catplot(data = df_unif)
        plt.title("uniform RC hyper-parameters")
        plt.show()
        
        
        sns.catplot(data = df_exp)
        plt.title("exponential RC hyper-parameters")
        plt.xticks(rotation=45)
        plt.show()

        sns.catplot(data = df_cyclic)
        plt.title("Cyclic RC hyper-parameters")
        plt.xticks(rotation=45)
        plt.show()

    
    def get_df(self):
        """
        this dataframe is essential for 
        """
        IGNORE_IP = False


        def quick_dirty_convert(lst, n):
            lst *= n
            pd_ = pd.DataFrame(np.array(lst).reshape(-1,1))
            return(pd_)

        for i, experiment in enumerate(self.experiment_lst):
            if not i:
                n_keys = len(list(experiment["nrmse"].keys()))
                idx_lst = list(range(len(self.experiment_lst)))
                idx_lst = quick_dirty_convert(idx_lst, n_keys)
                obs_hz_lst, targ_hz_lst, targ_freq_lst = [], [], []
            #print(experiment['experiment_inputs'].keys())
            targ_hz = experiment["experiment_inputs"]["target_hz"]
            obs_hz  = experiment["experiment_inputs"]["obs_hz"]
            targ_freq = experiment["experiment_inputs"]['target_frequency']

            if experiment["experiment_inputs"]["target_hz"] < 1:
                targ_hz *= 1000*1000
                obs_hz  *= 1000*1000
            obs_hz_lst  += [obs_hz]
            targ_hz_lst += [targ_hz]
            targ_freq_lst += [targ_freq]


            hz_line = {"target hz" : targ_hz }
            hz_line = Merge(hz_line , {"obs hz" : obs_hz })

            #print(hz_line)
            df_spec= experiment["nrmse"]

            #df_spec = Merge(experiment["nrmse"], {"target hz": targ_hz})
            df_spec = pd.DataFrame(df_spec, index = [0])

            df_spec_rel = df_spec.copy()
            #/df_spec_diff["uniform"]
            #df_spec_diff["rc_diff"]

            if IGNORE_IP == True:
                df_spec_rel = df_spec_rel / experiment["nrmse"]["uniform"]#
            else:
                try:
                    f_spec_rel = df_spec_rel / experiment["nrmse"]["ip: linear"]
                except:
                    f_spec_rel = df_spec_rel / experiment["nrmse"]["interpolation"]



            #print( df_spec_rel)
            #print(experiment["experiment_inputs"].keys())
            if i == 0:
                df      = df_spec
                df_rel  = df_spec_rel


            else:
                df = pd.concat([df, df_spec])
                df_rel = pd.concat([df_rel, df_spec_rel])


        df_net = df_rel.copy()

        obs_hz_lst, targ_hz_lst = quick_dirty_convert(obs_hz_lst, n_keys), quick_dirty_convert(targ_hz_lst, n_keys)
        targ_freq_lst = quick_dirty_convert(targ_freq_lst, n_keys)
        #display(df)
        if IGNORE_IP == True:
            df_rel = df_rel.drop(columns = ["interpolation"])
            df  = df.drop(columns = ["interpolation"])
        #df_rel  = df_rel.drop(columns = ["hybrid"])
        #df      = df.drop(    columns = ["hybrid"])

        df, df_rel = pd.melt(df), pd.melt(df_rel)
        df  = pd.concat( [idx_lst, df,  obs_hz_lst, targ_hz_lst, targ_freq_lst] ,axis = 1)

        df_rel = pd.concat( [idx_lst, df_rel,  obs_hz_lst, targ_hz_lst, targ_freq_lst], axis = 1)

        #df_diff = pd.concat( [idx_lst, df_diff,  obs_hz_lst, targ_hz_lst, targ_freq_lst], axis = 1)

        col_names = ["experiment", "model", "nrmse", "obs hz", "target hz", "target freq" ]
        df.columns, df_rel.columns    = col_names, col_names
        self.df, self.df_rel = df, df_rel

    def plot_nrmse_kde_2d(self, 
                          xx = "target hz", 
                          log = True, 
                          alph = 1, 
                          black_pnts = True, 
                          models = {"interpolation" : "Greens", "exponential" : "Reds", "uniform" : "Blues"},
                          enforce_bounds = False,
                          target_freq = None):
        """
        #todo description
        """
        if target_freq != None:
            df_spec = self.df[self.df["target freq"] == target_freq]
        else:
            df_spec = self.df.copy()
                
        
        def plot_(model_, colorr, alph = alph,  black_pnts =  black_pnts):
            if colorr == "Blues":
                color_ = "blue"
            elif colorr == "Reds":
                color_ = "red"
            elif colorr == "Greens":
                color_ = "green"
                
            df_ = df_spec[df_spec.model == model_] #df_ip  = df[df.model == "interpolation"]
            
            #display(df_)
                
            
            hi = df_["nrmse"]
            cap = 1
            if log == True:
                hi = np.log(hi)/ np.log(10)
                cap = np.log(cap) / np.log(10)
            
            
            sns.kdeplot(df_[xx], hi, cmap= colorr, 
                        shade=True, shade_lowest=False, ax = ax, label = model_, alpha = alph)#, alpha = 0.5)
            
            if  black_pnts == True:
                col_scatter = "black"
            else:
                col_scatter = color_
            
            sns.scatterplot(x = xx, y = hi, data = df_,  linewidth=0, 
                            color = col_scatter, alpha = 0.4, ax = ax)
            
            plt.title("2d kde plot: nrmse vs " + xx)
            
            plt.axhline(y=cap, color=color_, linestyle='-', label = "mean " + str(model_), alpha = 0.5)
            sns.lineplot(y = hi, x = xx, data = df_ , color = color_)#, alpha = 0.2)
            if enforce_bounds == True:
                ax.set_ylim(0,1)
            if log == True:
                ax.set_ylabel("log( NRMSE) ")
            else: 
                ax.set_ylabel("NRMSE")
                
        fig, ax = plt.subplots(1, 1, figsize = (12,6))
        for model in list(models.keys()):
            print(model)
            plot_(model, models[model], alph = alph)
        #plot_("interpolation", "Blues")
        #plot_("exponential", "Reds", alph = alph)
    
    def kde_plots(self,
                  target_freq = None, 
                  log = False, 
                  model = "uniform", 
                   models = {"ip: linear" : "Greens", "exponential" : "Reds", "uniform" : "Blues"},
                   enforce_bounds = True,
                   split = None):
        """
        HEATMAP EXAMPLE:
                         enforce_bounds = True)
        flights = flights.pivot("month", "year", "passengers") #y, x, z
        ax = sns.heatmap(flights)
        plot_nrmse_kde_2d(**additional_arguments, 
                          models = {"interpolation" : "Greens", "exponential" : "Reds", "uniform" : "Blues"})
        
        plot_nrmse_kde_2d(xx = "obs hz", **additional_arguments, 
                          models = {"interpolation" : "Greens", "exponential" : "Reds", "uniform" : "Blues"})
        """
        
        additional_arguments ={ "black_pnts" : False, 
                               "alph" : 0.3, 
                               "target_freq" : target_freq}    
        
        cmap = "coolwarm"
        
       
        def add_noise(np_array, log = log):
            sizee = len(np_array)
            x =  np.random.randint(100, size = sizee) + np_array 
            
            return(x)
        
        nrmse_dict = {}
        
        for i, model in enumerate(["uniform", "exponential", "ip: linear"]):#"interpolation"]):
            df_ = self.df[self.df.model == model ]
            
            xx, yy = add_noise(df_["target hz"]), add_noise(df_["obs hz"])

            nrmse= df_["nrmse"]
            if log == True:
                print("hawabunga")
                nrmse = np.log(nrmse)
            nrmse_dict[model] = nrmse
        
        
        
        """
        nrmse_diff = nrmse_dict["exponential"].values.reshape(-1,)  - nrmse_dict["uniform"].values.reshape(-1,) 
        print("(+): " + str(np.sum((nrmse_diff > 0)*1)))
        
        print("(-): " + str(np.sum((nrmse_diff < 0)*1)))
        
        
        display(nrmse_diff)
        xx, yy = add_noise(df_["target hz"]), add_noise(df_["obs hz"])
        #sns.distplot(nrmse_diff, ax = ax[2])
        sns.scatterplot(x = xx, y = yy, data = df_, ax = ax[2], palette=cmap, alpha = 0.9, s = 50, hue = nrmse_diff) #size = nrmse,
        ax[2].set_title(" diff: exponential - uniform" )
        plt.show()
        """
        
        self.plot_nrmse_kde_2d(**additional_arguments, log = False, 
                          models = models, #{"exponential" : "Reds", "uniform" : "Blues", "interpolation" : "Greens"},
                         enforce_bounds = True)
        
        
        self.plot_nrmse_kde_2d(xx = "obs hz", **additional_arguments, log = False, 
                           models = models, #{"exponential" : "Reds", "uniform" : "Blues", "interpolation" : "Greens"},
                           enforce_bounds = True)