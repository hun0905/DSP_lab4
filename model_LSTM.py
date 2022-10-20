from turtle import forward
from nltk.util import pr
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

import torch.nn.functional as F
from torch.utils.data import  DataLoader,random_split,SubsetRandomSampler
from sklearn.metrics import accuracy_score
class LSTM(nn.Module):
    def __init__(self,n_mels,input_size,hidden_size1,hidden_size2,num_layers1,num_layers2,output_size):#
        super(LSTM,self).__init__() #繼承父類nn.Module 的特性
        self.n_mels = n_mels
        self.lstm1 = nn.LSTM(input_size,hidden_size1,num_layers1,bidirectional = True,batch_first = True)
        self.lstm2 = nn.LSTM(hidden_size1*2,hidden_size2,num_layers2,bidirectional = True,batch_first = True)
        self.fc =  nn.Linear(hidden_size2*2*n_mels,output_size)
        self.dropout = nn.Dropout(p=0.05)
        self.criterion = torch.nn.CrossEntropyLoss() 
        self.class_num = output_size
    def forward(self,input):
        output,(hn,cn) = self.lstm1(input) #input : (batch_size , seq_len , embedding_dim)
        output = self.dropout(output)         
        output,(hn,cn) = self.lstm2(output) #input : (batch_size , seq_len , embedding_dim)
        output = self.dropout(output) 
        output = torch.reshape(output,(output.shape[0],-1))
        output = self.fc(output) 
        return output
    def fit(self,trainData=None,trainlabel=None,batch_size=32,use_cuda=True,device='cuda',epoch=20,optimizer=None):
        self.train()  #確保layers of model 在train mode
        train_loader = DataLoader(trainData, batch_size=batch_size)
        label_loader = DataLoader(trainlabel, batch_size=batch_size)
        total_loss = 0
        train_preds = [] #存放model預估的標點
        train_trues = [] #存放label的真實標點
        for e in range(epoch):
            for  i,(data) in enumerate(zip(train_loader,label_loader)):
                input , label = data#輸入的資料(文
                #字換成index,句子長度，label(標點的index)
                #print(segment)
                if  use_cuda:
                    input = input.cuda()
                    label = label.cuda()
                    input = input.to(device)
                    label = label.to(device)
                input = input.float()
                outputs = self.forward(input)#將資料輸入model(調用model的forward),outputs為評估結果
                #將outputs和label的dimension轉換，在用crossentropy評估loss
                outputs = outputs.view(-1, outputs.size(-1))
                if use_cuda:
                    outputs = outputs.to(device)
                label = label.view(-1)
                loss = self.criterion(outputs, label)
                loss.backward()#更新梯度
                optimizer.step() #計算weight
                optimizer.zero_grad() #將梯度清空
                total_loss += loss.item()
                train_outputs = outputs.argmax(dim=1) #outputs原輸出的是四種class的機率分佈,換成最高機率class的index
                train_preds.extend(train_outputs.detach().cpu().numpy())
                train_trues.extend(label.detach().cpu().numpy())
        accuracy = accuracy_score(train_trues, train_preds) 
        print("train accuracy: ",accuracy)
    def predict(self,test_data,batch_size=32,use_cuda=True,device='cuda'):
        val_loss = 0
        self.eval()
        #後面大致跟train epoch差不多
        val_preds = []
        val_trues = []
        test_loader = DataLoader(test_data, batch_size=batch_size)
        for i,(data) in enumerate(test_loader):
            input = data
            if  use_cuda:
                input = input.cuda()#換成可傳入gpu的型態
                input = input.to(device)     
            input = input.float()      
            outputs = self.forward(input)
            outputs = outputs.view(-1, outputs.size(-1))
            if use_cuda:
                outputs = outputs.to(device)
           
            val_outputs = outputs.argmax(dim=1)
            val_preds.extend(val_outputs.detach().cpu().numpy())
        return val_preds
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)########
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['n_mels'], package['input_size'], package['hidden_size1'],
                    package['hidden_size2'], package['num_layers1'],
                    package['num_layers2'], package['output_size'])
        model.load_state_dict(package['state_dict'])
        return model
        
    def serialize(self,model, optimizer,scheduler, epoch,train_loss,val_loss):
        package = {
            # hyper-parameter
            'n_mels': model.n_mels,
            'input_size': model.input_size,
            'hidden_size1': model.hidden_size1,
            'hidden_size2': model.hidden_size2,
            'num_layers1': model.num_layers1,
            'num_layers2': model.num_layers2,
            'output_size': model.output_size,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss':val_loss
        }
        return package

