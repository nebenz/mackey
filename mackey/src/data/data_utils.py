import os
import torch
import hydra
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# warnings.filterwarnings("ignore", message="The version_base parameter is not specified")

# ==================== Hyper-Params =================================================================================
EPS = 1e-8
# ===========================================================================================================

class DataPreProcessing():
    def __init__(self,args,device=None):
        self.args = args
        self.device = device
        self.dataUtills = DataUtills()

   


    
    #the output of this function should be ready to be send to the model
    def pre_processing(self, traj_raw):
        
        ########### Min - Max normalization  ###################

       
        min = traj_raw.min(axis=1)
        max = traj_raw.max(axis=1)
        
        traj_norm = (traj_raw - min)/ (max - min) 
        return traj_norm, min, max


class DataUtills():
    def __init__(self):
        super().__init__()
    
    #return the trajectory the the original scale 
    @staticmethod  
    def post_process(proocessed_traj,min,max):
        
        orig_traj  =  ((proocessed_traj)*(max-min))+min
        
        return orig_traj
        
   








        

