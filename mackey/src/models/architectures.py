import sys
import torch.nn as nn
import torch
import numpy as np
# from parts_model import *
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.utils import weight_norm
import math
import torchvision.models as models



epsilon = 1e-6


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, args, input_size=1, hidden_size= 30, output_size=1): #input size: num of channels 
        super(TCN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=4)
        self.conv4 = nn.Conv1d(hidden_size, output_size, kernel_size=3, padding=1, dilation=8)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(28, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc(x)
        return x
    
    
class TCN_with_Residual_Encoder(nn.Module):
    def __init__(self, args):#, input_size=1, hidden_size= 50, output_size=1,layers=5, kernel_s=5, dropout=0.1):
    # def __init__(self, args, nb_classes=1, Chans=1, Samples=500, layers=5, kernel_s=10, filt=40, dropout=0.1, activation='relu'):
        super(TCN_with_Residual_Encoder, self).__init__()
        
        self.args = args
        #layers = args.layers
        self.padding_1 =  (self.args.kernel_s - 1) * self.args.dilation_size 
         
        self.conv1 = nn.Conv1d(self.args.input_ch, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_1, dilation=self.args.dilation_size )
        self.batch_norm1 = nn.BatchNorm1d(self.args.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_1, dilation=self.args.dilation_size)
        
        dilation_3 = 2
        self.padding_2 = (self.args.kernel_s - 1) * dilation_3
        self.conv3 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_2, dilation=dilation_3)
        
        
        dilation_4 = 4
        self.padding_3 = (self.args.kernel_s - 1) * dilation_4
        self.conv4 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_3, dilation=dilation_4)
        
        dilation_5 = 8
        self.padding_4 = (self.args.kernel_s - 1) * dilation_5
        self.conv5 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_4, dilation=dilation_5)
        
        dilation_6 = 16
        self.padding_5 = (self.args.kernel_s - 1) * dilation_6
        self.conv6 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_5, dilation=dilation_6)
        
        
        self.cls = nn.Parameter(torch.ones(50))
        encod1 = nn.TransformerEncoderLayer(self.args.hidden_size, nhead=10,  batch_first=True, dropout=0.1, dim_feedforward=self.args.hidden_size*4)
        self.encoder = nn.TransformerEncoder(encoder_layer = encod1 , num_layers = 4)
        
        self.fc1 = nn.Linear(50,1)
        
        
        
        self.conv_skip_1 = nn.Conv1d(self.args.input_ch, self.args.hidden_size, kernel_size=1, padding='same')
        self.conv_skip_all = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=1, padding='same')
        
       
        
        
    def forward(self, x):
        
        #dilation 1
        out1 = self.conv1(x)

        out1 = out1[:,:,:-self.padding_1]
        out1 = self.relu(out1)
        out1 = self.batch_norm1(out1)
        out1 = self.dropout(out1)
        
        out2 = self.conv2(out1)
        out2 = out2[:,:,:-self.padding_1]
        out2 = self.relu(out2)
        out2 = self.batch_norm1(out2)
        out2 = self.dropout(out2)
        
        skip_out = self.conv_skip_1(x)
        block_1 = self.relu(torch.add(out2,skip_out))
        
        #dilation 2
         
        out3 = self.conv3(block_1)
        out3 = out3[:,:,:-self.padding_2]
        out3 = self.relu(out3)
        out3 = self.batch_norm1(out3)
        out3 = self.dropout(out3)
        
        out4 = self.conv3(out3)
        out4 = out4[:,:,:-self.padding_2]
        out4 = self.relu(out4)
        out4 = self.batch_norm1(out4)
        out4 = self.dropout(out4)
        
        skip_out = self.conv_skip_all(block_1)
        block_2 = self.relu(torch.add(out4,skip_out))
        
        #dilation 4
        
        out5 = self.conv4(block_2)
        out5 = out5[:,:,:-self.padding_3]
        out5 = self.relu(out5)
        out5 = self.batch_norm1(out5)
        out5 = self.dropout(out5)
        
        
        out6 = self.conv4(out5)
        out6 = out6[:,:,:-self.padding_3]
        out6 = self.relu(out6)
        out6 = self.batch_norm1(out6)
        out6 = self.dropout(out6)
        
        skip_out = self.conv_skip_all(block_2)
        block_3= self.relu(torch.add(out6,skip_out))
        
        #dilation 8
        
        out7 = self.conv5(block_3)
        out7 = out7[:,:,:-self.padding_4]
        out7 = self.relu(out7)
        out7 = self.batch_norm1(out7)
        out7 = self.dropout(out7)
        
        out8 = self.conv5(out7)
        out8 = out8[:,:,:-self.padding_4]
        out8 = self.relu(out8)
        out8 = self.batch_norm1(out8)
        out8 = self.dropout(out8)
        
        skip_out = self.conv_skip_all(block_3)
        block_4 = self.relu(torch.add(out8,skip_out))
        
        #dilation 16
        
        out9 = self.conv6(block_4)
        out9 = out9[:,:,:-self.padding_5]
        out9 = self.relu(out9)
        out9 = self.batch_norm1(out9)
        out9 = self.dropout(out9)
        
        out10 = self.conv6(out9)
        out10 = out10[:,:,:-self.padding_5]
        out10 = self.relu(out10)
        out10 = self.batch_norm1(out10)
        out10 = self.dropout(out10)
        
        #from here
        skip_out = self.conv_skip_all(block_4)
        block_5 = self.relu(torch.add(out10,skip_out))
        
        
        out11 = block_5.transpose(1,2)
        cls = self.cls.unsqueeze(0).repeat(out11.shape[0],1).unsqueeze(2).transpose(1,2)
        out12 = torch.cat((out11,cls), dim=1)
        out13 = self.encoder(out12)
        
        out  = self.fc1(out13[:,-1,:])
        
        
       
        return out


class TCN_with_Residual(nn.Module):
    def __init__(self, args):#, input_size=1, hidden_size= 50, output_size=1,layers=5, kernel_s=5, dropout=0.1):
    # def __init__(self, args, nb_classes=1, Chans=1, Samples=500, layers=5, kernel_s=10, filt=40, dropout=0.1, activation='relu'):
        super(TCN_with_Residual, self).__init__()
        
        self.args = args
        #layers = args.layers
        self.padding_1 =  (self.args.kernel_s - 1) * self.args.dilation_size 
         
        self.conv1 = nn.Conv1d(self.args.input_ch, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_1, dilation=self.args.dilation_size )
        self.batch_norm1 = nn.BatchNorm1d(self.args.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_1, dilation=self.args.dilation_size)
        
        dilation_3 = 2
        self.padding_2 = (self.args.kernel_s - 1) * dilation_3
        self.conv3 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_2, dilation=dilation_3)
        
        
        dilation_4 = 4
        self.padding_3 = (self.args.kernel_s - 1) * dilation_4
        self.conv4 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_3, dilation=dilation_4)
        
        dilation_5 = 8
        self.padding_4 = (self.args.kernel_s - 1) * dilation_5
        self.conv5 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_4, dilation=dilation_5)
        
        dilation_6 = 16
        self.padding_5 = (self.args.kernel_s - 1) * dilation_6
        self.conv6 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_5, dilation=dilation_6)
        
        
        
        self.conv_skip_1 = nn.Conv1d(self.args.input_ch, self.args.hidden_size, kernel_size=1, padding='same')
        self.conv_skip_all = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=1, padding='same')
        
        self.cnov_last = nn.Conv1d(self.args.hidden_size, self.args.input_ch, kernel_size=1)
        self.fc1 = nn.Linear(self.args.window_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        
        #dilation 1
        out1 = self.conv1(x)

        out1 = out1[:,:,:-self.padding_1]
        out1 = self.relu(out1)
        out1 = self.batch_norm1(out1)
        out1 = self.dropout(out1)
        
        out2 = self.conv2(out1)
        out2 = out2[:,:,:-self.padding_1]
        out2 = self.relu(out2)
        out2 = self.batch_norm1(out2)
        out2 = self.dropout(out2)
        
        skip_out = self.conv_skip_1(x)
        block_1 = self.relu(torch.add(out2,skip_out))
        
        #dilation 2
         
        out3 = self.conv3(block_1)
        out3 = out3[:,:,:-self.padding_2]
        out3 = self.relu(out3)
        out3 = self.batch_norm1(out3)
        out3 = self.dropout(out3)
        
        out4 = self.conv3(out3)
        out4 = out4[:,:,:-self.padding_2]
        out4 = self.relu(out4)
        out4 = self.batch_norm1(out4)
        out4 = self.dropout(out4)
        
        skip_out = self.conv_skip_all(block_1)
        block_2 = self.relu(torch.add(out4,skip_out))
        
        #dilation 4
        
        out5 = self.conv4(block_2)
        out5 = out5[:,:,:-self.padding_3]
        out5 = self.relu(out5)
        out5 = self.batch_norm1(out5)
        out5 = self.dropout(out5)
        
        
        out6 = self.conv4(out5)
        out6 = out6[:,:,:-self.padding_3]
        out6 = self.relu(out6)
        out6 = self.batch_norm1(out6)
        out6 = self.dropout(out6)
        
        skip_out = self.conv_skip_all(block_2)
        block_3= self.relu(torch.add(out6,skip_out))
        
        #dilation 8
        
        out7 = self.conv5(block_3)
        out7 = out7[:,:,:-self.padding_4]
        out7 = self.relu(out7)
        out7 = self.batch_norm1(out7)
        out7 = self.dropout(out7)
        
        out8 = self.conv5(out7)
        out8 = out8[:,:,:-self.padding_4]
        out8 = self.relu(out8)
        out8 = self.batch_norm1(out8)
        out8 = self.dropout(out8)
        
        skip_out = self.conv_skip_all(block_3)
        block_4 = self.relu(torch.add(out8,skip_out))
        
        #dilation 16
        
        out9 = self.conv6(block_4)
        out9 = out9[:,:,:-self.padding_5]
        out9 = self.relu(out9)
        out9 = self.batch_norm1(out9)
        out9 = self.dropout(out9)
        
        out10 = self.conv6(out9)
        out10 = out10[:,:,:-self.padding_5]
        out10 = self.relu(out10)
        out10 = self.batch_norm1(out10)
        out10 = self.dropout(out10)
        
        skip_out = self.conv_skip_all(block_4)
        block_5 = self.relu(torch.add(out10,skip_out))
        
        
        out11 = self.relu(self.cnov_last(block_5))
        flatten = out11.view(out11.shape[0], -1)
       
        out = self.relu(self.fc1(flatten))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
       
        
        
     
        return out



class TCN_with_Residual_Deepen(nn.Module):
    def __init__(self, args):#, input_size=1, hidden_size= 50, output_size=1,layers=5, kernel_s=5, dropout=0.1):
    # def __init__(self, args, nb_classes=1, Chans=1, Samples=500, layers=5, kernel_s=10, filt=40, dropout=0.1, activation='relu'):
        super(TCN_with_Residual_Deepen, self).__init__()
        
        self.args = args
        #layers = args.layers
        self.padding_1 =  (self.args.kernel_s - 1) * self.args.dilation_size 
         
        self.conv1 = nn.Conv1d(self.args.input_ch, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_1, dilation=self.args.dilation_size )
        self.batch_norm1 = nn.BatchNorm1d(self.args.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_1, dilation=self.args.dilation_size)
        
        dilation_3 = 2
        self.padding_2 = (self.args.kernel_s - 1) * dilation_3
        self.conv3 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_2, dilation=dilation_3)
        
        
        dilation_4 = 4
        self.padding_3 = (self.args.kernel_s - 1) * dilation_4
        self.conv4 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_3, dilation=dilation_4)
        
        dilation_5 = 8
        self.padding_4 = (self.args.kernel_s - 1) * dilation_5
        self.conv5 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_4, dilation=dilation_5)
        
        dilation_6 = 16
        self.padding_5 = (self.args.kernel_s - 1) * dilation_6
        self.conv6 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_5, dilation=dilation_6)
        
        dilation_7 = 32
        self.padding_6 = (self.args.kernel_s - 1) * dilation_7
        self.conv7 = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_6, dilation=dilation_7)
        
        dilation_8 = 64
        self.padding_7 = (self.args.kernel_s - 1) * dilation_8
        self.conv8= nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=self.args.kernel_s, padding=self.padding_7, dilation=dilation_8)
        
        
        
        self.conv_skip_1 = nn.Conv1d(self.args.input_ch, self.args.hidden_size, kernel_size=1, padding='same')
        self.conv_skip_all = nn.Conv1d(self.args.hidden_size, self.args.hidden_size, kernel_size=1, padding='same')
        
        self.cnov_last = nn.Conv1d(self.args.hidden_size, self.args.input_ch, kernel_size=1)
        self.fc1 = nn.Linear(self.args.window_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        
        #dilation 1
        out1 = self.conv1(x)

        out1 = out1[:,:,:-self.padding_1]
        out1 = self.relu(out1)
        out1 = self.batch_norm1(out1)
        out1 = self.dropout(out1)
        
        out2 = self.conv2(out1)
        out2 = out2[:,:,:-self.padding_1]
        out2 = self.relu(out2)
        out2 = self.batch_norm1(out2)
        out2 = self.dropout(out2)
        
        skip_out = self.conv_skip_1(x)
        block_1 = self.relu(torch.add(out2,skip_out))
        
        #dilation 2
         
        out3 = self.conv3(block_1)
        out3 = out3[:,:,:-self.padding_2]
        out3 = self.relu(out3)
        out3 = self.batch_norm1(out3)
        out3 = self.dropout(out3)
        
        out4 = self.conv3(out3)
        out4 = out4[:,:,:-self.padding_2]
        out4 = self.relu(out4)
        out4 = self.batch_norm1(out4)
        out4 = self.dropout(out4)
        
        skip_out = self.conv_skip_all(block_1)
        block_2 = self.relu(torch.add(out4,skip_out))
        
        #dilation 4
        
        out5 = self.conv4(block_2)
        out5 = out5[:,:,:-self.padding_3]
        out5 = self.relu(out5)
        out5 = self.batch_norm1(out5)
        out5 = self.dropout(out5)
        
        
        out6 = self.conv4(out5)
        out6 = out6[:,:,:-self.padding_3]
        out6 = self.relu(out6)
        out6 = self.batch_norm1(out6)
        out6 = self.dropout(out6)
        
        skip_out = self.conv_skip_all(block_2)
        block_3= self.relu(torch.add(out6,skip_out))
        
        #dilation 8
        
        out7 = self.conv5(block_3)
        out7 = out7[:,:,:-self.padding_4]
        out7 = self.relu(out7)
        out7 = self.batch_norm1(out7)
        out7 = self.dropout(out7)
        
        out8 = self.conv5(out7)
        out8 = out8[:,:,:-self.padding_4]
        out8 = self.relu(out8)
        out8 = self.batch_norm1(out8)
        out8 = self.dropout(out8)
        
        skip_out = self.conv_skip_all(block_3)
        block_4 = self.relu(torch.add(out8,skip_out))
        
        #dilation 16
        
        out9 = self.conv6(block_4)
        out9 = out9[:,:,:-self.padding_5]
        out9 = self.relu(out9)
        out9 = self.batch_norm1(out9)
        out9 = self.dropout(out9)
        
        out10 = self.conv6(out9)
        out10 = out10[:,:,:-self.padding_5]
        out10 = self.relu(out10)
        out10 = self.batch_norm1(out10)
        out10 = self.dropout(out10)
        
        skip_out = self.conv_skip_all(block_4)
        block_5 = self.relu(torch.add(out10,skip_out))
        
        #dilation 32
        
        out11 = self.conv7(block_5)
        out11 = out11[:,:,:-self.padding_6]
        out11 = self.relu(out11)
        out11 = self.batch_norm1(out11)
        out11 = self.dropout(out11)
        
        out12 = self.conv7(out11)
        out12 = out12[:,:,:-self.padding_6]
        out12 = self.relu(out12)
        out12 = self.batch_norm1(out12)
        out12 = self.dropout(out12)
        
        skip_out = self.conv_skip_all(block_5)
        block_6 = self.relu(torch.add(out12,skip_out))
        
        #dilation 64
        
        out13 = self.conv8(block_6)
        out13 = out13[:,:,:-self.padding_7]
        out13 = self.relu(out13)
        out13 = self.batch_norm1(out13)
        out13 = self.dropout(out13)
        
        out14 = self.conv8(out13)
        out14 = out14[:,:,:-self.padding_7]
        out14 = self.relu(out14)
        out14 = self.batch_norm1(out14)
        out14 = self.dropout(out14)
        
        
        
        skip_out = self.conv_skip_all(block_6)
        block_7 = self.relu(torch.add(out14,skip_out))
        
        
        out15 = self.relu(self.cnov_last(block_7))
        flatten = out15.view(out15.shape[0], -1)
       
        out = self.relu(self.fc1(flatten))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
       
        
        
     
        return out



