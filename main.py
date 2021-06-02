import tensorflow as tf
import numpy as np

import pickle
import random
import argparse
from pathlib import Path

import utils
from dataset import EMNIST_Data, Omniglot_Data, Shakespeare_Data, Stackoverflow_Data, CIFAR100_Data
import models




parser = argparse.ArgumentParser()


## Dataset options

    #    Datasource selects the dataset upon which the model will be trained
    #    --datasource => omniglot, miniimagenet, cifar10, cifar100, fashion-mnist, emnist
parser.add_argument("-ds", "--datasource", dest="datasource", type=str, default='emnist')
parser.add_argument("-g", "--gpu_num", dest="gpu_num", type=int, default=0)
parser.add_argument("-s", "--shots", dest="shots", type=int, default=5)
parser.add_argument("-w", "--ways", dest="ways", type=int, default=5)

## Training options

    #    Method selects the methodology upon which the model will be trained
    #    --method => hf2meta, jiang, fedmeta, perfed, federated
parser.add_argument("-m", "--method", type=str, dest="method", default="jiang")
parser.add_argument("-ms", "--meta_step_size", type=float, dest="meta_step_size", default=0.25)
parser.add_argument("-lr", "--learning_rate", dest="learning_rate", type=float, default=0.1)
parser.add_argument("-ir", "--inner_rounds", dest="inner_rounds", type=int, default=5)
parser.add_argument("-cr", "--comm_rounds", dest="comm_rounds", type=int, default=500)
parser.add_argument("-tr", "--test_rounds", dest="test_rounds", type=int, default=1)

## Environment options
parser.add_argument("-n", "--num_clients", dest="num_clients", type=int, default=10)
parser.add_argument("-sd", "--server_data", dest="server_data", type=int, default=0)
parser.add_argument("-uc", "--use_clients", dest="use_clients", type=bool, default=True)

parser.add_argument("-r", "--rounds", dest="rounds", type=int, default=1)

## Logging, saving, and testing options
parser.add_argument("-v", "--verbose", dest='verbose', type=bool, default=True)
parser.add_argument("-t", "--train", dest="train", type=bool, default=True)

args = parser.parse_args()



def main():
    logdir = f'./logs/{args.datasource}/{args.method}'
    accdir = f'{logdir}/{args.num_clients}clients_accs.pkl'
    lossdir = f'{logdir}/{args.num_clients}clients_loss.pkl'
    moddir = f'./models/{args.datasource}/{args.method}/model_weights.pkl'

    Path(logdir).mkdir(parents=True, exist_ok=True)
    Path(moddir).mkdir(parents=True, exist_ok=True)

    # Tensorflow GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[args.gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[args.gpu_num], True)

    # Dataset settings
    if args.datasource == 'omniglot' or args.datasource == 'emnist':
        input_dim = (28, 28, 1)
    else:
        input_dim = (32, 32, 3)
    use_server_data = args.server_data > 0

    accs = list()
    losses = list()
    for i in range(args.rounds):
        tf.random.set_seed(random.randint(1, 10000))
        # Create model

        if args.datasource == 'emnist':
            dataset = EMNIST_Data(server_prop = args.server_data)
        elif args.datasource == 'cifar100':
            dataset = CIFAR100_Data(server_prop = args.server_data)
        elif args.datasource == 'shakespeare':
            dataset = Shakespeare_Data(server_prop= args.server_data)
        elif args.datasource == 'stackoverflow':
            dataset = Stackoverflow_Data(server_prop = args.server_data)
        elif args.datasource == 'omniglot':
            dataset = Omniglot_Data(server_proportion= args.server_data, classes= args.ways, way= args.shots)

        if args.datasource == 'omniglot' or args.datasource == 'fashion_mnist' or args.datasource == 'emnist' or args.datasource == 'mnist':
            model = models.cnn(classes=dataset.classes, input_dim=input_dim)
        elif args.datasource == 'cifar100':
            model = models.resnet(classes=dataset.classes, input_dim=input_dim)
        elif args.datasource == 'shakespeare':
            model = models.lstm_shakespeare(len(dataset.vocab))
        elif args.datasource == 'stackoverflow':
            model = models.lstm_stack(dataset.classes)

            
        #use_server=use_server_data, server_proportion=args.server_data, use_clients=args.use_clients, num_clients=args.num_clients, datasource=args.datasource, input_dim=input_dim

        if args.train == True:
            print(f"Initializing {args.method} methodology")
            if args.method == 'hf2meta':
                from lib.hf2meta import hf2_meta
                method = hf2_meta(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, True, args.use_clients, args.verbose, input_dim, logdir)

            elif args.method == 'jiang':
                from lib.reptile import Reptile
                method = Reptile(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, use_server_data, args.use_clients, args.verbose, input_dim, logdir)
            
            elif args.method == 'fedmeta':
                from lib.fedmeta import fedmeta
                method = fedmeta(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, use_server_data, args.use_clients, args.verbose, input_dim, logdir)

            elif args.method == 'perfed':
                from lib.perfedfo import perfed
                method = perfed(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, use_server_data, args.use_clients, args.verbose, input_dim, logdir)
            
            elif args.method == 'federated':
                from lib.federated import federated
                method = federated(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, use_server_data, args.use_clients, args.verbose, input_dim, logdir)

            elif args.method == 'hf2meta_fo':
                from lib.hf2meta_fo import hf2_meta
                method = hf2_meta(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, use_server_data, args.use_clients, args.verbose, input_dim, logdir)

            elif args.method == 'hf2meta1':
                from lib.hf2meta1 import hf2_meta
                method = hf2_meta(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, use_server_data, args.use_clients, args.verbose, input_dim, logdir)

            elif args.method == 'hf2meta2':
                from lib.hf2meta2 import hf2_meta
                method = hf2_meta(model, dataset, args.meta_step_size, args.learning_rate, args.inner_rounds, args.comm_rounds, args.test_rounds, args.num_clients, args.shots, args.ways, use_server_data, args.use_clients, args.verbose, input_dim, logdir)
        else:
            with open(moddir, 'rb') as f:
                model_weights = pickle.load(f)
            model.set_weights(model_weights)
            
            acc, std, loss = utils.evaluate_model(model, dataset, args.shots * args.ways, args.shots, args.ways, args.test_rounds, 100)

            print('test acc= {:.4f} +- {:.4f}% \n test loss: {:.4f}'.format(acc * 100, std * 100, loss))    
        
        print("Starting training")
        new_model, acc, loss = method.train()
        accs.append(acc)
        losses.append(loss)
    acc = np.mean(accs, axis=0)
    loss = np.mean(losses, axis=0)
    print("Finished training!\n Evaluating model & saving")

        # If first time, save the model
    method.save_accs(acc, accdir)
    method.save_loss(loss, lossdir)
    method.save_weights(new_model, moddir)
        


if __name__ == '__main__':
    main()