import argparse
from nn import FeedForwardNeuralNetwork
from losses import *
from optimizers import *
from learn import *
import wandb
from pathlib import Path
import os
import numpy as np
from sklearn.model_selection import train_test_split

args_parser = argparse.ArgumentParser()

args_parser.add_argument('-wp', '--wandb_project', default=None, help='Name of the wandb project to track run on')
args_parser.add_argument('-we', '--wandb_entity', default=None, help='Name of the wandb user')
args_parser.add_argument('-d', '--dataset', type=str.lower, choices=['mnist', 'fashion_mnist'], default='fashion_mnist', help='Dataset to evaluate on')
args_parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train for')
args_parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size used during training')
args_parser.add_argument('-l', '--loss', type=str.lower, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy', help='Loss function to use for optimization')
args_parser.add_argument('-o', '--optimizer', type=str.lower, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='adam', help='Optimization algorithm to use')
args_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate used by the optimizer')
args_parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum used by momentum and nag optimizers')
args_parser.add_argument('-beta', '--beta', type=float, default=0.9, help='Beta used by rmsprop')
args_parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers')
args_parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers')
args_parser.add_argument('-eps', '--epsilon', type=float, default=1e-8, help='Epsilon used for numerical stability in calculations')
args_parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help='Weight decay/L2-regularization used by the optimizers')
args_parser.add_argument('-w_i', '--weight_init', type=str.lower, choices=['random', 'xavier'], default='xavier', help='Weight initialization method used for the model')
args_parser.add_argument('-nhl', '--num_layers', type=int, default=4, help='Number of hidden layers in the model')
args_parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in each hidden layer of the model')
args_parser.add_argument('-a', '--activation', type=str.lower, choices=['identity', 'sigmoid', 'tanh', 'relu'], default='relu', help='Activation function used in the input and hidden layers of the model')
args_parser.add_argument('-v', '--verbose', type=int, default=2, help='Verbosity of information printed during training. verbose <= 0: No information is printed. verbose = 1: Information at the end of training is printed. verbose >= 2: Information is printed during each epoch of training')
args_parser.add_argument('-run', '--run_test', action='store_true', help='Pass flag if trained model is to be run on test data')
args_parser.add_argument('-cm', '--conf_mat', action='store_true', help='Pass flag if confusion matrix (on test data) is to be logged with wandb. Ignored if --wandb_project is None or --run_test is not passed')
# add other args, like clip_norm (if needed)

args = args_parser.parse_args()

DATA_DIR = Path('./data/')

# Get the required dataset
if args.dataset == 'fashion_mnist':
    data_path = Path('./data/fashion_mnist.npz')
    if data_path.is_file():
        print('Data found. Loading...')
        data = np.load(data_path)
        X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    else:
        print('Data not found. Downloading...')
        if not DATA_DIR.is_dir():
            os.mkdir(DATA_DIR)
        from keras.datasets import fashion_mnist
        (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
        X_train, X_test = X_train/255., X_test/255.
        np.savez_compressed(data_path, X_train, Y_train, X_test, Y_test)
    CLASS_LABELS = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]
elif args.dataset == 'mnist':
    data_path = Path('./data/mnist.npz')
    if data_path.is_file():
        print('Data found. Loading...')
        data = np.load(data_path)
        X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    else:
        print('Data not found. Downloading...')
        if not DATA_DIR.is_dir():
            os.mkdir(DATA_DIR)
        from keras.datasets import mnist
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        X_train, X_test = X_train/255., X_test/255.
        np.savez_compressed(data_path, X_train, Y_train, X_test, Y_test)
    CLASS_LABELS = [str(i) for i in range(10)]

# Construct the neural network as per the specifications
model = FeedForwardNeuralNetwork(
    num_layers=args.num_layers,
    hidden_size=args.hidden_size,
    activation=args.activation,
    weight_init=args.weight_init
)

# Get the loss function
if args.loss == 'cross_entropy':
    loss_fn = CrossEntropyLoss()
elif args.loss == 'mean_squared_error':
    loss_fn = MeanSquaredErrorLoss()

# Construct the required optimizer with given specifications
if args.optimizer == 'sgd':
    optimizer = StochasticGradientDescent(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
elif args.optimizer == 'momentum':
    optimizer = MomentumGradientDescent(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.momentum
    )
elif args.optimizer == 'nag':
    optimizer = NesterovGradientDescent(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.momentum
    )
elif args.optimizer == 'rmsprop':
    optimizer = RMSProp(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        beta=args.beta,
        eps=args.epsilon
    )
elif args.optimizer == 'adam':
    optimizer = Adam(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon
    )
elif args.optimizer == 'nadam':
    optimizer = NAdam(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon
    )

# Metric for classification
metric = categorical_accuracy

# Train-validation split
X_t, X_v, Y_t, Y_v = train_test_split(X_train.reshape(-1, 784), Y_train, test_size=0.1)

# Train the model
history = fit(
    model=model,
    X_train=X_t,
    Y_train=Y_t,
    X_val=X_v,
    Y_val=Y_v,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metric=metric,
    epochs=args.epochs,
    batch_size=args.batch_size,
    verbose=args.verbose
)

# Test the model (if necessary)
if args.run_test:
    test_preds = predict(model, X_test.reshape(-1, 784))
    test_accuracy = metric(Y_test, test_preds)
    print(f'Accuracy on test data: {test_accuracy}')

# Log run data with wandb (if necessary)
if args.wandb_project is not None:
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity) as run:
        run.name = f'{args.dataset}_{args.epochs}epochs_bs_{args.batch_size}_{args.loss}_{args.optimizer}_lr_{args.learning_rate}_wdecay_{args.weight_decay}_{args.weight_init}_hl_{args.num_layers}_hs_{args.hidden_size}_{args.activation}' 
        for i in range(args.epochs):
            wandb.log({'epoch': history['epoch'][i], 'loss': history['train_loss'][i], 'accuracy': history['train_score'][i], 'val_loss': history['val_loss'][i], 'val_accuracy': history['val_score'][i]})
        if args.conf_mat:
            wandb.log({'confusion_matrix': wandb.plot.confusion_matrix(probs=test_preds, y_true=Y_test, class_names=CLASS_LABELS)})
