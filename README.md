# CS6910-A1
Assignment 1 of CS6910 - Fundamentals of Deep Learning

## Outline & contents

The neural network framework is split into multiple scripts, named as follows:
- [**activations.py**](./activations.py): Contains a generic Activation class, which is subclassed to implement the following activation functions:
    1. Identity
    2. Sigmoid
    3. Tanh
    4. ReLU
    5. Softmax 
- [**losses.py**](./losses.py): Contains a generic LossFunction class, which is subclassed to implement the following loss functions:
    1. Cross-entropy loss
    2. Mean squared error loss
- [**optimizers.py**](./optimizers.py): Contains a generic Optimizer class, which is subclassed to implement the following optimizers:
    1. Stochastic Gradient Descent
    2. Momentum Gradient Descent
    3. Nesterov Accelerated Gradient Descent
    4. RMSProp
    5. Adam
    6. NAdam
- [**metrics.py**](./metrics.py): Implements the necessary scoring metrics as functions. Only categorical accuracy is implemented as it is the only metric required for the assignment.
- [**nn.py**](./nn.py): Implements the backbone neural network in two levels, via the Layer class and the FeedForwardNeuralNetwork class, which uses multiple `Layer` classes to implement a network-level functionality. 
- [**learn.py**](./learn.py): Contains functions (fit, predict) to train a neural network and to predict outputs using a neural network.

Two other helper scripts are also present:
- [**train.py**](./train.py): A high-level abstraction, using the code framework described above, to train a neural network on the specified dataset (MNIST/Fashion-MNIST) for the described configuration and quantify its performance, with the ability to track the run on wandb.
- [**sweep.py**](./sweep.py): Performs a sweep on wandb, using the configuration specified in the script. Only used for code separation and not an essential part otherwise for the assignment.

The notebook [**A1.ipynb**](./A1.ipynb) was used for checking the working of the code along the way and only contains abstract usages and tests for the most part.

## Running the code

The train script **train.py** can be run as follows:
```
python3 train.py [-h] [-wp WANDB_PROJECT] [-we WANDB_ENTITY] [-d {mnist,fashion_mnist}] [-e EPOCHS] [-b BATCH_SIZE] [-l {mean_squared_error,cross_entropy}] [-o {sgd,momentum,nag,rmsprop,adam,nadam}] [-lr LEARNING_RATE] [-m MOMENTUM] [-beta BETA] [-beta1 BETA1] [-beta2 BETA2] [-eps EPSILON] [-w_d WEIGHT_DECAY] [-w_i {random,xavier}] [-nhl NUM_LAYERS] [-sz HIDDEN_SIZE] [-a {identity,sigmoid,tanh,relu}] [-v VERBOSE] [-run] [-cm]
```

The available options, in detail, are as follows:
| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | None | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | None  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 10 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 16 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | momentum | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.01 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.9 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 1e-8 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0.0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | xavier | choices:  ["random", "xavier"] | 
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |
| `-v`, `--verbose` | 2 | Verbosity level of information printed during training (<=0, 1, >=2). |
| `-run`, `--run_test` | N/A | Computes performance on test data if passed |
| `-cm`, `--conf_mat` | N/A | Logs confusion matrix on wandb if passed |

To run the code on a particular configuration, execute
```
python3 train.py [-wp WANDB_PROJECT] [-we WANDB_ENTITY] [-run (if running on test data)] [-cm (if logging test confusion matrix on wandb)] {configuration} 
```

## Extending the code 

A new optimizer, loss function, activation function or a scoring metric can easily be added to extend the functionality implemented by adding a new entry in the corresponding script file.
- A new optimizer can be implemented in **optimizers.py** by subclassing `Optimizer` and implementing the `step` method
- A new loss function can be implemented in **losses.py** by subclassing `LossFunction` and implementing the `forward` and `backward` methods, to compute the loss (scalar) and to compute the gradient of the computed loss wrt the inputs to the loss function 
- A new activation function can be implemented in **activations.py** by subclassing `Activation` and implementing the `forward` and `backward` methods, to compute the activated values and to compute the gradient of the activation outputs wrt the activation inputs (given the outputs)
- A new scoring metric can be implemented in **metrics.py** as a function, which accepts the true labels and the predicted probabilities as inputs and computes the score