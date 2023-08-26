import numpy as np
import torch
class mnist_dataset():
    def __init__(self,dataset):
        self.class_number = len(dataset.classes)
        self.classes = dataset.classes
        self.data,self.targets =self.reshape(dataset.data),dataset.targets
        self.channels, self.width, self.height = self.__shape_info__()
   
    def __shape_info__(self):
        return self.data.shape[1:]
    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]
        label = np.zeros(10)
        label[target] = 1 
        label = torch.Tensor(label) 
        return image,label
    def __len__(self):
        return len(self.data)
    def reshape(self,data):
        new_data = data.reshape(len(data),1,28,28)
        return np.array(new_data)
