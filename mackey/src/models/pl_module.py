import sys
import pytorch_lightning as pl
import random, os

# sys.path.append('/workspace/inputs/asaf')
from architectures import TCN,  TCN_with_Residual, TCN_with_Residual_Deepen, TCN_with_Residual_Encoder



import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

class load_pl_module(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,x):
        pred = self.model(x)
        return pred

class load_models():
    def __init__(self,args):
        super().__init__()
        self.args = args
     

    @staticmethod
    def choose_model(args): 
       
       
        if args.model_name=='TCN_with_Residual_Encoder':
            return  TCN_with_Residual_Encoder(args)
        elif args.model_name=='TCN_with_Residual_Deepen':
            return  TCN_with_Residual_Deepen(args)
        elif args.model_name=='TCN':
            return  TCN(args)
        elif args.model_name=='TCN_with_Residual':
            return  TCN_with_Residual(args)
        
       
 
    def load(self,model_def_name):
        model_type = self.choose_model(self.args)
        model = load_pl_module(model_type)
        model = model.load_from_checkpoint(model=model_type,checkpoint_path =self.args.infer_from_ckpt)
        return model
