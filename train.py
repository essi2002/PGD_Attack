from model import BadNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch

def train(train_loader):
    epochs = 100
    model = BadNet(input_channels=1,output_num=len(train_loader.dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    for epoch in range(epochs):
        current_loss = 0
        model.train()
        for step,(batch_idx,batch_idy) in enumerate(tqdm(train_loader)):
             optimizer.zero_grad()
             output = model(batch_idx)

             loss = criterion(output,batch_idy)
            
             loss.backward()
             optimizer.step()
             current_loss += loss
        print("Epoch%d loss :%.4f" %(epoch,current_loss))
    return model
        
        
