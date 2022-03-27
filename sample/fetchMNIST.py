import os
import numpy as np
import requests, gzip, os, hashlib

path = r'./MNIST'

if not os.path.isdir(path):
	os.mkdir(path)
	print(f'create directory {path}')

def fetch(url, filename):
    #fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    fp = os.path.join(path, filename)
    if not os.path.isfile(fp):
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)

    #return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "X_train")#[0x10:].reshape((-1, 28, 28))
fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "y_train")#[8:]
fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "X_test")#[0x10:].reshape((-1, 28*28))
fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "y_test")#[8:]
print("successfully fetch MNIST hand written data")
