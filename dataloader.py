import numpy as np
from PIL import Image

class ImgFileDataloader:
    def __init__(self,file_list,label_array,batch_size=32,shape=(28,28)):
        self.file_list = file_list
        self.length = len(self.file_list)
        self.label_array = label_array
        self.order = self.choose_random_order()
        self.index = 0
        self.batch_size = batch_size
        self.shape=shape
    def choose_random_order(self):
        self.order = np.random.choice(np.arange(len(self.file_list)),len(self.file_list),replace=False)
        return self.order
    def read_img(self,file):
        return np.array(Image.open(file))/255.
    def data_generator(self):
        while True:
            x = np.zeros((self.batch_size,self.shape[0],self.shape[1],1),dtype=np.float32)
            file_list = self.file_list[self.order[self.index:self.index+self.batch_size]]
            for i,file in enumerate(file_list):
                x[i,:,:,:] = self.read_img(file).reshape(1,self.shape[0],self.shape[0],1)
            y = np.eye(10)[self.label_array[self.order[self.index:self.index+self.batch_size]]]
            x = x[0:y.shape[0]]
            if self.index + self.batch_size >= self.length:
                self.choose_random_order()
                self.index = 0
            else:
                self.index += self.batch_size

            yield x,y