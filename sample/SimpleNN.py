import numpy as np
import pandas as pd
import time

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
            return grad_input, (grad_weights, grad_biases)

        def update(self, update, lr = 0.01):
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

    class CrossEntropyLoss:

        def __init__(self) -> None:
            pass

        def get_grad(self, y_out, y_true):

            # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
            onehot_answers = np.zeros_like(y_out)
            onehot_answers[np.arange(len(y_out)), y_true.astype(np.int32)] = 1
            
            softmax = np.exp(y_out) / np.exp(y_out).sum(axis=-1,keepdims=True)

            return (- onehot_answers + softmax) / y_out.shape[0]
        
        def get_loss(self, y_out, y_true):
            prob_answers = y_out[np.arange(len(y_out)),y_true.astype(np.int32)]
            xentropy = - prob_answers + np.log(np.sum(np.exp(y_out),axis=-1))
        
            return xentropy.sum()

        def __str__(self) -> str:
            return "CrossEntropyLoss"

    class MSELoss:

        def __init__(self) -> None:
            pass

        def get_grad(self):
            pass

        def get_loss(self):
            pass

        def __str__(self) -> str:
            return "MSELoss"

    def __init__(self, layers_config) -> None:
        
        self.layers = []
        self.loss_func = None

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

            # cross entropy loss
            elif layer_list[0] == 'CrossEntropy':
                self.loss_func = self.CrossEntropyLoss()

            # MSE loss
            elif layer_list[0] == 'MSELoss':
                self.loss_func = self.CrossEntropyLoss()

        # gradients
        self.grads = None

        # trian flag (allow gradients or not)
        self._train = True
        self._lr = 0

        self._log = dict(zip(['train_acc', 'train_loss', 'valid_acc', 'valid_loss'], [None]*4))

    def summary(self) -> None:
        
        print('_'*30)
        for layer in self.layers:
            print(layer)
            print('_'*30)

    def save(self):

        for layer in self.layers:
            pass

    def _reset_log(self) -> None:
        self._log['train_acc'] = list()
        self._log['valid_acc'] = list()
        self._log['train_loss'] = list()
        self._log['valid_loss'] = list()

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
        loss_grad = self.loss_func.get_grad(out, targets)
        # loss_grad = 2*(out-targets)/out.shape[0]

        for layer_index in reversed(range(len(self.layers))):
            loss_grad, update = self.layers[layer_index].backward(self.layer_inputs[layer_index], loss_grad)
            self.layers[layer_index].update(update, self._lr)

    def fit(self, X_train, y_train, X_valid = None, y_valid = None, epochs = 100, lr = 1e-3):
        
        self._train = True
        self._lr = lr
        self._reset_log()
        training_time = 0

        if isinstance(X_valid, pd.DataFrame) or isinstance(X_valid, np.ndarray):
            valid_flag = True

        #targets =  np.apply_along_axis(lambda a: np.array([0 if i != a else 1 for i in range(int(max(y_train))+1)]), 1, y_train.reshape((y_train.shape[0], 1)))
        targets = y_train.copy()

        # train for n epoch
        for epoch in range(epochs):

            start_time = time.time()

            # train part
            y = self.forward(X_train)
            self.backward(y, targets)

            category=np.argmax(y,axis=1)
            train_loss = self.loss_func.get_loss(y, y_train)

            train_accuracy = (category == y_train).mean()
            self._log['train_acc'].append(train_accuracy)
            self._log['train_loss'].append(train_loss)

            # calculate how many time an epoch take
            # note only train time are take into account
            epoch_time = (time.time() - start_time) * 1000 # milisecond
            training_time += epoch_time

            # validate part
            if valid_flag: 
                y = self.forward(X_valid)
                category = np.argmax(y, axis=1)
                valid_loss = self.loss_func.get_loss(y, y_valid)
                valid_accuracy = (category == y_valid).mean()
                self._log['valid_acc'].append(valid_accuracy)
                self._log['valid_loss'].append(valid_loss)
                print(f'epoch {epoch+1}/{epochs} | train accuracy: {train_accuracy:.2f} | valid accuracy: {valid_accuracy:.2f}\ntrain loss: {train_loss:.3f} | valid loss: {valid_loss:.3f} | epoch time {epoch_time:.2f} ms')
            else:
                print(f'epoch {epoch+1}/{epochs} | train accuracy: {train_accuracy:.2f} | trian loss: {train_loss:.3f} | epoch time {epoch_time:.2f} ms')
            print('_'*70)

        best_train_epoch = np.argmax(self._log['train_acc'])
        best_train_acc = self._log['train_acc'][best_train_epoch]
        best_train_loss = self._log['train_loss'][best_train_epoch]
        print(f'best train acc: {best_train_acc:.2f}, loss = {best_train_loss:.3f}')
        if valid_flag:
            best_valid_epoch = np.argmax(self._log['valid_acc'])
            best_valid_acc = self._log['valid_acc'][best_valid_epoch]
            best_valid_loss = self._log['valid_loss'][best_valid_epoch]
            print(f'best valid acc: {best_valid_acc:.2f}, loss = {best_valid_loss:.3f}')
        print(f'average training time {training_time/epochs:.3f}')

    def softmax_crossentropy_with_logits(self,logits,reference_answers):
        # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
        logits_for_answers = logits[np.arange(len(logits)),reference_answers.astype(np.int32)]
        
        xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
        
        return xentropy
    
    def grad_softmax_crossentropy_with_logits(self,logits,reference_answers):
        # Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers
        # ones_for_answers = np.zeros_like(logits)
        # ones_for_answers[np.arange(len(logits)),reference_answers] = 1
        
        softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)

        return (- reference_answers + softmax) / logits.shape[0]
