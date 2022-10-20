from turtle import forward
from nltk.util import pr
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import  DataLoader,random_split,SubsetRandomSampler
from sklearn.metrics import accuracy_score
class CNN_classifier(nn.Module):
    def __init__(self,input_channel,height,width,output_size):#
        super(CNN_classifier,self).__init__() #繼承父類nn.Module 的特性
        #input (input_cha*nnel,height,width)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channel, 300,(14,10), 1, 0),  # (48,height,width)
            nn.ReLU(),
            # nn.MaxPool2d((2,1), 2, 0),      # (48,height/2,width/2)
            nn.Conv2d(300, 250, (12,8), 1, 0), # (256,height/2,width/2)
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # (256,height/4,width/4)
            nn.Conv2d(250, 250, (8,6), 1, 0), # (192,height/4,width/4)
            nn.ReLU(),
            nn.Conv2d(250, 150, (6,4), 1, 0), # (192,height/4,width/4)
            nn.ReLU(),
            nn.Conv2d(150, 50, (4,2), 1, 0), # (192,height/4,width/4)
            nn.ReLU(),
        )
        #we set the number of filters as the number of out_channels
        self.fc = nn.Sequential(
            nn.Linear(int(450), 5),
        )
        self.criterion = torch.nn.CrossEntropyLoss() 
        self.class_num = output_size
    def forward(self,input):
        out = self.cnn(input)
        # print(np.shape(out))
        out = out.view(out.size()[0], -1)
        return self.fc(out)
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
        model = cls(package['input_channel'], 
        package['height'],
        package['width'],
        package['output_size'])
        model.load_state_dict(package['state_dict'])
        return model
        
    def serialize(self,model, optimizer,scheduler, epoch,train_loss,val_loss):
        package = {
            # hyper-parameter
            'input_size': model.input_channel,
            'height': model.height,
            'width': model.width,
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

