import time
import numpy as np
import matplotlib.pyplot as plt
import requests, gzip, os, hashlib

from collections import Counter

class FNN():

    class Dense:

        def __init__(self, input_size, output_size, activation) -> None:
            self._weight = np.random.uniform(-1., 1., size = (input_size, output_size)) / np.sqrt(input_size * output_size).astype(np.double)
            self._biases = np.zeros(output_size)

            self._out = 0

            if activation == 'sigmoid':
                self._activate = FNN.Sigmoid()
            elif activation == 'softmax':
                self._activate = FNN.Softmax()
            elif activation == 'relu':
                self._activate = FNN.ReLu()
            elif activation == 'n':
                self._activate = FNN.Identical()
            else:
                raise RuntimeError("unknow activation function used")

        def forward(self, x):
            self._out = x.dot(self._weight) + self._biases
            return self._activate.forward(self._out)

        def backward(self, x_input, grad_output):
            
            grad_output = grad_output * self._activate.df_dx(self._out)
            grad_input = np.dot(grad_output, self._weight.T)
            # grad_input = (self._weight.dot(grad_output.T)).T
            # _error = _error.dot(self._weight.T)
            
            grad_weights = np.dot(x_input.T, grad_output)
            grad_biases = grad_output.mean(axis=0)*x_input.shape[0]

            self._out = 0
            # print(abs(grad_weights).sum())
            return grad_input, (grad_weights, grad_biases)

        def update(self, update, lr = 0.01):
            # print(update.sum())
            self._weight -= update[0] * lr
            self._biases -= update[1] * lr

        def __str__(self) -> str:
            return f"Linear Layer {self._weight.shape} {self._activate}"

    class Identical:

        def __init__(self) -> None:
            return

        def forward(self, x):
            return x
        
        def df_dx(self, x):
            return 1

        def __str__(self) -> str:
            return "None"

    class Sigmoid:

        def __init__(self) -> None:
            return

        def forward(self, x):
            return 1/(np.exp(-x)+1)

        def df_dx(self, x):
            return (np.exp(-x))/((np.exp(-x)+1)**2)

        def __str__(self) -> str:
            return "Sigmoid"

    class Softmax:

        def __init__(self) -> None:
            return

        def forward(self, x):
            exponents=np.exp(x)
            return exponents/np.sum(exponents)

        def df_dx(self, x):
            exp_element=np.exp(x-x.max())
            return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))


        def __str__(self) -> str:
            return "Softmax"

    class ReLu:

        def __init__(self) -> None:
            return

        def forward(self, x):
            return np.maximum(0, x)
        
        def df_dx(self, x):
            return np.where(x>0, 1, 0)

        def __str__(self) -> str:
            return "ReLu"

    def __init__(self, layers_config) -> None:
        
        self.layers = []

        # build layers
        for layer in layers_config.split('\n'):
            layer_list = layer.strip().split(',')

            # detect empty layer command
            if len(layer_list) <=0:
                raise RuntimeError("empty layer")

            # add linear layer
            if layer_list[0] == 'linear':

                try:
                    input_size = int(layer_list[1])
                    output_size = int(layer_list[2])
                    activation = layer_list[3].strip()
                except:
                    raise RuntimeError('invalid argument for linear layer')

                self.layers.append(self.Dense(input_size=input_size, output_size=output_size, activation=activation))

        # gradients
        self.grads = None

        # trian flag (allow gradients or not)
        self._train = True
        self._lr = 0

    def summary(self) -> None:
        
        print('_'*30)
        for layer in self.layers:
            print(layer)
            print('_'*30)

    def forward(self, x):

        _x = x.copy()

        if self._train:
            self.layer_inputs = []
            self.layer_inputs.append(_x.copy())

        # feedforward
        for layer in self.layers:
            _x = layer.forward(_x)
            
            # require gradients
            if self._train:
                self.layer_inputs.append(_x.copy())

        return _x

    def backward(self, out, targets):


        # loss = self.softmax_crossentropy_with_logits(out, targets)
        loss_grad = self.grad_softmax_crossentropy_with_logits(out, targets)
        # loss_grad = 2*(out-targets)/out.shape[0]

        for layer_index in reversed(range(len(self.layers))):
            loss_grad, update = self.layers[layer_index].backward(self.layer_inputs[layer_index], loss_grad)
            self.layers[layer_index].update(update, self._lr)

    def train(self, X_train, y_train, X_valid = None, y_valid = None, epochs = 100, lr = 1e-3):
        
        self._train = True
        self._lr = lr

        targets = targets = np.apply_along_axis(lambda a: np.array([0 if i != a else 1 for i in range(10)]), 1, y_train.reshape((y_train.shape[0], 1)))
        train_accuracies, valid_accuracies = [], []

        # train for n epoch
        for epoch in range(epochs):

            start_time = time.time()

            # train part
            y = model.forward(X_train)
            model.backward(y, targets)

            category=np.argmax(y,axis=1)
            train_loss = self.softmax_crossentropy_with_logits(y, y_train).sum()

            train_accuracy = (category == y_train).mean()
            train_accuracies.append(train_accuracy)

            # calculate many time a epoch cost
            # note only train time are take into account
            epoch_time = time.time() - start_time

            # validate part
            if X_valid.any() != None and y_valid.any() != None: 
                y = model.forward(X_valid)
                category = np.argmax(y, axis=1)
                valid_loss = self.softmax_crossentropy_with_logits(y, y_valid).sum()
                valid_accuracy = (category == y_valid).mean()
                valid_accuracies.append(valid_accuracy)


            if X_valid.any() != None and y_valid.any() != None: 
                print(f'epoch {epoch+1}/{epochs} | train accuracy: {train_accuracy:.2f} | valid accuracy: {valid_accuracy:.2f}\ntrain loss: {train_loss:.3f} | valid loss: {valid_loss:.3f} | epoch time {epoch_time:.2f}')
            else:
                print(f'epoch {epoch+1}/{epochs} | train accuracy: {train_accuracy:.2f} | trian loss: {train_loss:.3f} | epoch time {epoch_time:.2f}')
            print('_'*70)

    def softmax_crossentropy_with_logits(self,logits,reference_answers):
        # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
        logits_for_answers = logits[np.arange(len(logits)),reference_answers]
        
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
        
        return xentropy
    
    def grad_softmax_crossentropy_with_logits(self,logits,reference_answers):
        # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
        # ones_for_answers = np.zeros_like(logits)
        # ones_for_answers[np.arange(len(logits)),reference_answers] = 1
        
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
        
        return (- reference_answers + softmax) / logits.shape[0]

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
