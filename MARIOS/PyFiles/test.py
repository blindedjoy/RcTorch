import json
def load_data(file = "default"):
    if file == "default":
        nf = get_new_filename(exp = exp, current = True)
    else:
        nf = file
    with open(nf) as json_file: # 'non_exp_w.txt'
        datt = json.load(json_file)
    #datt = non_exp_best_args["dat"]
    #datt["obs_tr"], datt["obs_te"]   = np.array(datt["obs_tr"]), np.array(datt["obs_te"])
    #datt["resp_tr"], datt["resp_te"] = np.array(datt["resp_tr"]), np.array(datt["resp_te"])
    return(datt)

#experiment.save_json(exp = False)
print("DATA STRUCTURE: (it's a dict)")
bp =  "./"
fp = bp + 'experiment_results/2k/medium/split_0.5/targetKhz:_0.01__obskHz:_0.03.txt'
hi = load_data(file = fp)
for i in hi.keys():
    print(i + "/")
    
    if type(hi[i]) == dict:
        
        for j in hi[i].keys():
            print("    " +j)
print("/n inputs:")
print(hi["experiment_inputs"])
print(hi["get_observer_inputs"])

print("/n key saved values:")
print(hi["best arguments"])
print(hi["nrmse"])