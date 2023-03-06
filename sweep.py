import wandb
from sklearn.model_selection import train_test_split
from nn import FeedForwardNeuralNetwork
from losses import *
from optimizers import *
from metrics import categorical_accuracy
from learn import *
from pathlib import Path
import os
import datetime as dt
import argparse

args_parser = argparse.ArgumentParser()

args_parser.add_argument('-l', '--loss', type=str.lower, choices=['cross_entropy', 'mean_squared_error'], help='Loss function to use for training the model')

args = args_parser.parse_args()

data_dir = Path('./data/')
data_path = Path('./data/fashion_mnist.npz')

if data_path.is_file():
    print('Data found. Loading...')
    data = np.load(data_path)
    X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
else:
    print('Data not found. Downloading...')
    if not data_dir.is_dir():
        os.mkdir(data_dir)
    from keras.datasets import fashion_mnist
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train, X_test = X_train/255., X_test/255.
    np.savez_compressed(data_path, X_train, Y_train, X_test, Y_test)

def perform_run(config=None):
    with wandb.init(config=config) as run:
        config = wandb.config
        run.name = f'{"MSE" if args.loss=="mean_squared_error" else "CE"}_hl_{config.num_layers}_hs_{config.hidden_size}_{config.activation[:4]}_{config.weight_init[:3]}_{config.optimizer}_lr_{config.lr}_wd_{config.weight_decay}_bs_{config.batch_size}'
        X_t, X_v, Y_t, Y_v = train_test_split(X_train.reshape(-1, 784), Y_train, test_size=0.1)
        model = FeedForwardNeuralNetwork(config.num_layers, config.hidden_size, activation=config.activation, weight_init=config.weight_init)
        loss_fn = CrossEntropyLoss() if args.loss == 'cross_entropy' else MeanSquaredErrorLoss()
        if config.optimizer == 'sgd':
            optimizer = StochasticGradientDescent(model, lr=config.lr, weight_decay=config.weight_decay)
        elif config.optimizer == 'momentum':
            optimizer = MomentumGradientDescent(model, lr=config.lr, weight_decay=config.weight_decay)
        elif config.optimizer == 'nag':
            optimizer = NesterovGradientDescent(model, lr=config.lr, weight_decay=config.weight_decay)
        elif config.optimizer == 'rmsprop':
            optimizer = RMSProp(model, lr=config.lr, weight_decay=config.weight_decay)
        elif config.optimizer == 'adam':
            optimizer = Adam(model, lr=config.lr, weight_decay=config.weight_decay)
        elif config.optimizer == 'nadam':
            optimizer = NAdam(model, lr=config.lr, weight_decay=config.weight_decay)
        metric = categorical_accuracy
        history = fit(
            model,
            X_t,
            Y_t,
            X_v,
            Y_v,
            optimizer,
            loss_fn,
            metric,
            epochs=config.epochs,
            batch_size=config.batch_size,
            verbose=0
        )
        for i in range(len(history['epoch'])):
            wandb.log({'epoch': history['epoch'][i], 'loss': history['train_loss'][i], 'accuracy': history['train_score'][i], 'val_loss': history['val_loss'][i], 'val_accuracy': history['val_score'][i]})

sweep_config = {
    'method': 'random',
    'name' : f'{"MSE" if args.loss=="mean_squared_error" else "CE"}_'+str(dt.datetime.now().strftime("%d-%m-%y_%H:%M"))
}
sweep_metric = {
    'name': 'val_accuracy',
    'goal': 'maximize'
}
sweep_config['metric'] = sweep_metric
parameters = {
    'num_layers': {
        'values': [1, 2, 3, 4]
    },
    'hidden_size': {
        'values': [16, 32, 64, 128]
    },
    'activation': {
        'values': ['sigmoid', 'tanh', 'relu']
    },
    'weight_init': {
        'values': ['random', 'xavier']
    },
    'optimizer': {
        'values': ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']
    },
    'lr': {
        'values': [1e-1, 1e-2, 1e-3, 1e-4]
    },
    'weight_decay': {
        'values': [0.0, 1e-1, 1e-2, 1e-3]
    },
    'batch_size': {
        'values': [16, 32, 64, 128]
    }
}
sweep_config['parameters'] = parameters
parameters.update({
    'epochs': {
        'value': 10
    }
})
sweep_id = wandb.sweep(sweep_config, project='CS6910-A1') 

wandb.agent(sweep_id, perform_run, count=250)
