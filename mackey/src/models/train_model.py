import os
import sys
import hydra
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import hydra
import matplotlib.pyplot as plt
import numpy as np

print("__file__.split(src/)[0]:", __file__, __file__.split("src/")[0])
# sys.path.append(os.getcwd())
sys.path.append(__file__.split("src/")[0])

from src.data.Dataloader import  DataloaderModule
# from models.pl_module import seed_everything
import glob
from pl_module import load_models
from utills_model import pred_model
# seed_everything(42)

CRITERION = 'nn.criterion'

# ======================== Model section ===========================

class Pl_module(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = load_models.choose_model(args)
        
        self.y = []
        self.pred = []
       
     
        self.criterion = getattr(nn, self.args.criterion)()# (reduction= self.args.criterion_paramter_value)
        self.validation_step_outputs_loss = []
        # self.validation_step_outputs_accuracy = []
       
        

    def forward(self,batch): 
        pred_x = self.model(batch)
        return pred_x
    
    def infer(self, batch):
        x, _ = batch
        x = torch.unsqueeze(torch.unsqueeze(x,0),0)
        x = x.to(self.device)     
        y_hat = self(x)

        
        return y_hat

    def training_step(self, batch, batch_idx):
        whole_traj_batch = batch
        whole_traj_batch = whole_traj_batch.float()
        window_size = self.args.window_size
        prediction_step =  self.args.prediction_step
        traj_length = whole_traj_batch.shape[2]
        
        
        
        for i in range(0,traj_length-window_size-prediction_step,prediction_step): #TODO fit only in case step is 1 !!! 
            current_segment = whole_traj_batch[:,:,i:i+window_size]
            pred_next_step = self(current_segment).squeeze()
            true_val = whole_traj_batch[:,:,i+window_size+prediction_step]
            loss = self.Loss(pred_next_step.unsqueeze(1), true_val)
            self.log('train_loss', loss)
            
        #TODO aggregate trhe loss 
        return loss
        
    def validation_step(self, batch, batch_idx,loader_idx=None):
        whole_traj_batch = batch
        whole_traj_batch = whole_traj_batch.float()
        window_size = self.args.window_size
        prediction_step =  self.args.prediction_step
        traj_length = whole_traj_batch.shape[2]
        
        
        
        
        for i in range(0,traj_length-window_size-prediction_step,prediction_step): #TODO fit only in case step is 1 !!! 
            pred_next_step = self(whole_traj_batch[:,:,i:i+window_size]).squeeze()
            true_val = whole_traj_batch[:,:,i+window_size+prediction_step]
            loss = self.Loss(pred_next_step.unsqueeze(1), true_val)
            self.log('val_loss', loss, on_step = False, on_epoch=True)
            
        #TODO aggregate trhe loss 
        return loss
        
      
    
    def validation_epoch_end(self,batch):
        #run of test dataset
        loss = pred_model(self.args, self)
        self.log('test_loss',loss)
      

    def test_step(self, batch, batch_idx):
        pass
        # pred = self(batch)
        # loss = self.Loss(pred,y)
        # self.log('test_loss', loss)

    def configure_optimizers(self):

        # return torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=10, cooldown=3, verbose=True)
        return {'optimizer': optimizer,'scheduler':scheduler}

    def Loss(self,pred,y, parameters = None):

        return self.criterion(pred,y)

   
# ======================================== main section ==================================================

def find_existing_ckpt():

    folder_name = os.getcwd()
    ckpts = glob.glob(folder_name + "/**/last.ckpt", recursive=True)
    if len(ckpts) == 0:
        print("No checkpoints found.")
        return None
    times = [os.path.getctime(ckpt) for ckpt in ckpts]
    max_time = max(times)
    latest_index = times.index(max_time)
    print(f"Resuming from checkpoint: {ckpts[latest_index]}")
    return ckpts[latest_index]



Hydra_path = os.path.join(os.getcwd(), "src","config")
 # INTERACTIVE
# @hydra.main(config_path= Hydra_path,config_name="train.yaml")

# TRAIN
@hydra.main(config_path= __file__.split("src/")[0]+"src/config/",config_name="train.yaml", version_base='1.1')



def main(args):

# ====================== load config params from hydra ======================================

    pl_checkpoints_path = os.getcwd() + '/'
    # pl_checkpoints_path = args.models_dir + '/'


    if args.debug_flag: # debug mode
        a=1
    #     #fast_dev_run=True
    #     args.num_workers = 0 
    #     args.train_batch_size = 2
    #     args.val_batch_size = 2
    #     os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # else: # training mode
    #     fast_dev_run=False
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

# ============================= main section =============================================

    model = Pl_module(args) #load_models( )
    # model.load(args.model_name)

    device = 'cuda'# if torch.cuda.is_available() else 'cpu'
    # 'cuda'
    model= model.to(device)

    # existing_ckpt = find_existing_ckpt()
    # print(existing_ckpt)
    # if existing_ckpt:
    #     model = model.load_from_checkpoint(
    #         model=model,
    #         checkpoint_path=existing_ckpt, args=args)
    
    #  else:

    #     model = model.load_from_checkpoint(
    #         model=model,
    #         checkpoint_path=args.resume_from_checkpoint, cfg=args

    checkpoint_callback = ModelCheckpoint(
        monitor = args.ckpt_monitor,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_last = args.save_last,
        save_top_k = args.save_top_k,
        mode='min',
        verbose=True,
    )

    earlystopping_callback = EarlyStopping(monitor = args.ckpt_monitor, 
                                            patience = args.patience)

    trainer = Trainer(  
                        gpus=1, #args.gpus, 
                        accelerator = 'gpu',
                        strategy = DDPStrategy(find_unused_parameters=False),
                        fast_dev_run=False, 
                        check_val_every_n_epoch=args.check_val_every_n_epoch, 
                        default_root_dir= pl_checkpoints_path,                       
                        callbacks=[earlystopping_callback, checkpoint_callback], 
                        precision=32,
                        num_sanity_val_steps = 0,
                        max_epochs=1000
                     )
    
    data_module = DataloaderModule(args)
    trainer.fit(model = model, datamodule = data_module)
    checkpoint_callback.best_model_path

if __name__ == '__main__':
    main()
