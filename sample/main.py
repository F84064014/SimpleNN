import time
import numpy as np
import matplotlib.pyplot as plt
import requests, gzip, os, hashlib

from collections import Counter
from SimpleNN import FNN

if __name__ == '__main__':

    def load_MNIST(path):

        if os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
        
        return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

    X = load_MNIST(r"./MNIST/X_train")[0x10:].reshape((-1, 28, 28))
    Y = load_MNIST(r"./MNIST/y_train")[8:]
    X_test = load_MNIST(r"./MNIST/X_test")[0x10:].reshape((-1, 28*28))
    Y_test = load_MNIST(r"./MNIST/y_test")[8:]

    #Validation split
    rand=np.arange(60000)
    np.random.shuffle(rand)
    train_no=rand[:50000]

    val_no=np.setdiff1d(rand,train_no)

    X_train,X_val=X[train_no,:,:],X[val_no,:,:]
    Y_train,Y_val=Y[train_no],Y[val_no]

    with open('config', 'r') as fin:
        layer_config = fin.read()

    model = FNN(layers_config=layer_config)
    model.summary()

    X_train = X_train.reshape((50000, 28*28))
    X_val = X_val.reshape((X_val.shape[0], 28*28))
    
    model.train(X_train, Y_train, X_val, Y_val, lr = 0.07)
