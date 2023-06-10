import os
import torch
# from scipy.io.wavfile import write
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import hydra
import pickle
import torch.nn
import random



from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
sys.path.append(os.getcwd())
from src.data.data_utils import DataPreProcessing, DataUtills
import torch.nn as nn


class Metrics():
    def __init__(self):
        super().__init__()


    
    @staticmethod
    def post_process(args,pred,target):
        target = target.to(pred.device)
        bias = float(target - pred)
        SE = np.square(bias)
        MAE = np.abs(bias)

        score = {'bias': bias, 'SE': SE, 'MAE': MAE}
        # pred = pred[:,0,:,:]+1j*pred[:,1,:,:]
        # if args.cut_shape_f:
        #     pred = torch.cat((pred[:,0,:].unsqueeze(1)*0,pred),axis=1)
        # pred = torch.istft(pred,window=torch.hamming_window(args.window_size).to(pred.device),hop_length=args.overlap, n_fft=args.window_size,onesided=True,center=False)
        return bias, SE,  MAE


    def tests(self,args,pred,batch):

        return 
    
def create_fig(args, idx, traj_orig, traj_est):
  
  
 
    x1 = np.arange(0 , traj_orig.shape[1])
    x2 = np.arange(args.window_size, args.window_size + traj_est.shape[0])
    plt.figure(figsize=(16,9))
    plt.plot(x1,traj_orig[0,:],'b-.')
    plt.plot(x2,traj_est,'r-*') 
    folder_path = os.getcwd() + '/figures/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    plt.savefig(folder_path + '%d.png' % idx)
    
    
        
    
   
   

#for a single trajectory
def inference(pl_model,args, input, min, max):
    
    #TODO post process -  reset the data to the original range
    whole_traj = torch.from_numpy(input)
    whole_traj = whole_traj.float()
    whole_traj = whole_traj.to(pl_model.device)
    window_size = args.window_size
    prediction_step =  args.prediction_step
    traj_length = whole_traj.shape[1]
    loss_arr =  []
    loss = nn.MSELoss()
    real_dat = []
    est_dat = []
        
        
        
    for i in range(0,traj_length-window_size-prediction_step,prediction_step):
        current_segment = whole_traj[:,i:i+window_size]
        pred_next_step = pl_model(current_segment.unsqueeze(0)).squeeze(-1)
        true_val = whole_traj[:,i+window_size+prediction_step]
        loss_arr.append(loss(pred_next_step.squeeze(-1), true_val))
        est_dat.append(pred_next_step)
        # real_dat.append(true_val)
    
   
    return  torch.stack(loss_arr).mean(),est_dat
 

def pred_model(args,pl_model):

    # ============================= initialize =============================================
    
    # orig_wd =hydra.utils.get_original_cwd()
    files_path = args.data_dir.test
    
    with open(files_path, "rb") as file:
        
            file = pickle.load(file)
            data = file['data']        


    data = data.reshape(data.shape[0],data.shape[-1],data.shape[1])
    data_pre_process = DataPreProcessing(args)
    
    
   
    metrics = Metrics()
    score = {'mean_loss':[], 'pred':[]}
    # my_array = list(range(1000))
    random_ints = random.sample(list(range(data.shape[0])), 10)
    for traj_idx in  random_ints:

    # ============================= data pre process =============================================
        traj =  np.expand_dims(data[traj_idx,0,:], axis=0)
        traj_norm, min, max = data_pre_process.pre_processing(traj) 
        
        
        # ============================= inference =============================================
        # input_features = torch.unsqueeze(input_features,0)
        loss_arr, est_data = inference(pl_model,args, traj_norm, min, max)
        cat_est = torch.cat(est_data, dim=0)
        est_arr = cat_est.detach().cpu().numpy()
        traj_post_process = DataUtills.post_process(est_arr, min, max)
        if args.create_figs:
            create_fig(args,traj_idx, traj, traj_post_process)
        

    # ============================= post process ==================================================

        score['mean_loss'].append(loss_arr)
   
    
    return  torch.stack(score['mean_loss']).mean()

