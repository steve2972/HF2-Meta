import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import numpy as np
import time
import pickle

import utils
import models

class Reptile:
    def __init__(self, model, dataset, meta_step_size, learning_rate, 
                inner_rounds, comm_rounds, test_rounds, num_clients,
                shots, classes, server_data: bool, use_clients: bool, 
                verbose:bool, input_dim=(32,32,3), logdir='', p_sample = 0.1):
        self.model = model
        self.data = dataset
        self.mss = meta_step_size
        self.lr = learning_rate
        self.inner_rounds = inner_rounds
        self.comm_rounds = comm_rounds
        self.test_rounds = test_rounds
        self.num_clients = num_clients
        self.use_server_data = server_data
        self.use_clients = use_clients
        self.verbose = verbose
        self.inner_batch_size = shots * classes
        self.classes = classes
        self.shots = shots
        self.input_dim=input_dim
        self.logdir = logdir

        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']
        

        if self.data.datasource == 'omniglot' or self.data.datasource == 'fashion_mnist' or self.data.datasource == 'emnist' or self.data.datasource == 'mnist':
            self.global_model = models.cnn(classes=dataset.classes, input_dim=input_dim)
            self.optimizer = keras.optimizers.Adam(learning_rate=0.01)
            self.use_logits = False
            
        elif self.data.datasource == 'cifar100':
            self.global_model = models.resnet(classes=self.data.classes, input_dim=self.input_dim)
            self.optimizer = keras.optimizers.Adam()
            self.use_logits = False

        elif self.data.datasource == 'shakespeare':
            self.global_model = models.lstm_shakespeare(len(self.data.vocab))
            self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.use_logits = True
        
        elif self.data.datasource == 'stackoverflow':
            self.global_model = models.lstm_stack(dataset.classes)
            self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.use_logits = False
            
        
        # Create local model for clients
        self.model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)
        # Create a global model deployed on the server
        self.global_model.compile(optimizer=self.optimizer, loss=loss, metrics=metrics)

    def train(self):
        global_acc = list()
        global_loss = list()
        if self.verbose:
            with open(f'{self.logdir}/{self.num_clients}nc_{self.lr}lr_{self.mss}ms_{self.comm_rounds}cr_{self.inner_rounds}ir_{self.data.get_server_proportion()}server_prop', 'a+') as f:
                f.write('round,acc,loss\n')
                f.close()

        if self.use_server_data:
            for client in self.data.list_server:
                #server_data = self.data.get_server_data(self.inner_batch_size, self.inner_rounds, self.shots, self.classes, _+1)
                if self.data.datasource == 'shakespeare' and self.data.get_length_data(client, 4): 
                    server_data = self.data.get_server_data(client)
                    weights = self.train_client(self.model, server_data)
                    self.global_model.set_weights(weights)
                elif self.data.datasource != 'shakespeare':
                    server_data = self.data.get_server_data(client)
                    weights = self.train_client(self.model, server_data)
                    self.global_model.set_weights(weights)

            acc, val_loss = utils.test_model(self.global_model, 0, time.time(), self.data, self.test_rounds)

        for comm_round in range(self.comm_rounds):
            local_weights_list = list()

            global_weights = self.global_model.get_weights()
            cur_meta_step_size = (1-(comm_round/self.comm_rounds)) * self.mss

            start = time.time()
            ### Client-Side Calculations ###
            #print(f"Communication round {comm_round}")
            for idx, client in enumerate(np.random.choice(self.data.list_clients, size=self.num_clients, replace=False)):
                #print(f"\t-Training client{idx}")
                self.model.set_weights(global_weights)

                data = self.data.get_client_data(client)

                weights = self.train_client(self.model, data)

                grads = utils.weight_difference(weights, global_weights)

                local_weights_list.append(grads)

                K.clear_session()
            ### Server-Side Calculations ###

            
            # Average the gradients and update global parameters
            average_grads = utils.scale_model_weights(utils.sum_weights(local_weights_list), utils.weight_scaling_factor(self.num_clients)*-1)
            final_weights = utils.update_weights(global_weights, average_grads, cur_meta_step_size)
            self.global_model.set_weights(final_weights)

            if comm_round % 20 == 0:
                acc, val_loss = utils.test_model(self.global_model, comm_round+1, start, self.data, self.test_rounds)

                if self.verbose:
                    with open(f'{self.logdir}/{self.num_clients}nc_{self.lr}lr_{self.mss}ms_{self.comm_rounds}cr_{self.inner_rounds}ir_{self.data.get_server_proportion()}server_prop', 'a+') as f:
                        f.write('{:03d},{:.4f},{:.4f}\n'.format(comm_round, acc, val_loss))
                        f.close()
            else:
                print(f"Round: {comm_round+1} | elps: {time.time() - start:.2f}")


            global_acc.append(acc)
            global_loss.append(val_loss)

            K.clear_session()

        return self.global_model, global_acc, global_loss
    
    
    def train_client(self, model, data): 
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=self.use_logits)
        
        for epoch in range(self.inner_rounds):
            for step, (x_batch_train, y_batch_train) in enumerate(data):
                utils.train_on_batch(model, loss_fn, self.optimizer, x_batch_train, y_batch_train)

        return model.get_weights()

    def save_accs(self, accs, filename):
        with open(filename, 'wb') as f:
            pickle.dump(accs, f)

    def save_loss(self, loss, filename):
        with open(filename, 'wb') as f:
            pickle.dump(loss, f)

    def save_weights(self, model, filename):
        with open(filename, 'wb') as f:
            pickle.dump(model.get_weights(), f)