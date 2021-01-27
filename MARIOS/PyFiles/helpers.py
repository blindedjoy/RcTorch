

def Merge(dict1, dict2): 

    res = {**dict1, **dict2} 
    return res 

def nrmse(pred_, truth, columnwise = False):
    """
    inputs should be numpy arrays
    """
    if columnwise == True:
        rmse_ = np.sqrt((truth - pred_) ** 2)
        denom_ = np.sum(truth ** 2, axis = 1).reshape(-1, 1)
    else:
        rmse_ = np.sqrt(np.sum((truth - pred_) ** 2))
        denom_ = np.sum(truth ** 2)
    
    nrmse_ = rmse_ / denom_
    return(nrmse_)

def idx2Freq(val):
    idx = min(range(len(f)), key=lambda i: abs(f[i]-val))
    return(idx)