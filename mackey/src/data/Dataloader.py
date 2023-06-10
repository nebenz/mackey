import os
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import hydra
import pickle
import sys
sys.path.append(os.getcwd())
from src.data.data_utils import DataUtills, DataPreProcessing
import warnings
# from models.pl_module import seed_everything
# seed_everything(42)
import torch


warnings.filterwarnings("ignore", message="The version_base parameter is not specified")



class GeneretedInputOutput(Dataset):
    """Generated Input and Output to the model"""
    def __init__(self,args,data_dir):
        
       
        self.args = args
        a =1
        
        with open(data_dir, 'rb') as file:
             # Load the data from the pickle file
            self.data_dict =  pickle.load(file)
            self.keys = list(self.data_dict.keys())

        # for key in data:
        #     val = data[key]

        
        
        # with open(data_dir, "rb") as file:
        
        #     file = pickle.load(file)
        #     data = file['data']        

        # self.all_data = data
        # self.data = data.reshape(data.shape[0],data.shape[-1],data.shape[1])
        self.data_pre_process = DataPreProcessing(self.args)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        # ======== get file name ====================
        key = self.keys[idx]
        data_raw = self.data_dict[key]
        data_norm, _, _, = self.data_pre_process.pre_processing(data_raw)
        # Process and return the data
        return data_norm
        
        # traj_raw = self.data[idx]

        # # ==================== data-preprocessing ======================
        # traj_norm, _, _, = self.data_pre_process.pre_processing(traj_raw)
        # return traj_norm




   

class DataloaderModule(pl.LightningDataModule):
    # def __init__(self,args): 
    #     super().__init__()
    #     self.args=args


    def __init__(self, args,  batch_size=32, test_ratio=0.1, val_ratio=0.1, seed=42):
        super().__init__()
        
        self.args=args
        # self.dataset = dataset
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
    # def setup(self,stage=None):   
    #     self.train_set = GeneretedInputOutput(self.args,self.args.data_dir.train)
    #     self.val_set = GeneretedInputOutput(self.args,self.args.data_dir.val)
    #     self.test_set = GeneretedInputOutput(self.args,self.args.data_dir.test)
    
    def setup(self,stage=None):  
        torch.manual_seed(self.seed) 
        
        self.all_data =  GeneretedInputOutput(self.args,self.args.data_set_path)
        
          # Calculate the sizes for train, test, and validation sets
        dataset_size = len(self.all_data.keys)
        test_size = int(dataset_size * self.test_ratio)
        val_size = int(dataset_size * self.val_ratio)
        train_size = dataset_size - test_size - val_size

        # # Split the dataset into train, test, and validation sets
        self.train_set, self.val_set, self.test_set = random_split(
            self.all_data.keys, [train_size, val_size, test_size])
        
        # self.train_set = GeneretedInputOutput(self.args,self.args.data_dir.train)
        # self.val_set = GeneretedInputOutput(self.args,self.args.data_dir.val)
        # self.test_set = GeneretedInputOutput(self.args,self.args.data_dir.test)



    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.args.train_batch_size, shuffle=True , num_workers =self.args.num_workers, pin_memory= self.args.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.args.val_batch_size, shuffle= False, num_workers = self.args.num_workers, pin_memory= self.args.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.args.test_batch_size, shuffle= True, num_workers = self.args.num_workers, pin_memory= self.args.pin_memory)
            




Hydra_path = '/Users/Asaf/Dropbox/AI2C/Assignment/ml_assignment/exercise_4/mackey/src/config/'
@hydra.main(config_path= Hydra_path,config_name="train.yaml")
def main(args):
# #     # ============================= create instance model with regular Pl_module =============================================
    

# # #     '''
# # #     TODO:


    
    dm = DataloaderModule(args)
    dm.setup()
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    


    iterator_train = iter(train_dl)     
    batch_train = next(iterator_train)
   

    iterator_val = iter(val_dl)     
    batch_val = next(iterator_val)
   
    a =1

    

    



    

if __name__ == '__main__':
    main()