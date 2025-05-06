import numpy as np
from matplotlib import pyplot as plt
import scipy

from layers import (fully_connected_forward, fully_connected_backward, conv_with_padding_forward, 
                    conv_with_padding_backward, maxpooling_backward, maxpooling_forward, 
                    relu_backward, relu_forward, softmaxloss_backward, softmaxloss_forward)

def cifar10_starter():
    """
    Train a neural network on the CIFAR-10 dataset.

    Loads the CIFAR-10 dataset, preprocesses it, defines a neural network architecture,
    and trains the network using stochastic gradient descent. The trained model is then saved
    for future use, and its accuracy is evaluated on the test set.

    Returns:
    None
    """
    # Add relevant imports based on your specific implementation.
    
    # Load CIFAR-10 dataset
    base_path = "/your/path/here/"
    mat_file_path = base_path + "all_data.mat"
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: File '{mat_file_path}' not found.")
        mat_data = None

    if mat_data is not None:
        x_train = mat_data['x_train']
        x_train = x_train.reshape((32, 32, 3, 50000), order='F').astype(np.float32)
        y_train = mat_data['y_train'].reshape(-1) + 1
    mat_file_path = base_path + "cifar_test.mat"
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: File '{mat_file_path}' not found.")
        mat_data = None

    if mat_data is not None:
        x_test = mat_data['x_test']
        x_test = x_test.reshape((32, 32, 3, 10000), order='F').astype(np.float32)
        y_test = mat_data['y_test'].reshape(-1) + 1

    # Visualize images (optional)
    # Add visualization code here

    # Preprocess the data
    data_mean = np.mean(x_train)
    x_train -= data_mean
    x_test -= data_mean

    perm = np.random.permutation(len(y_train))
    x_train = x_train[:, :, :, perm]
    y_train = y_train[perm]

    x_val = x_train[:, :, :, -2000:]
    y_val = y_train[-2000:]
    x_train = x_train[:, :, :, :-2000]
    y_train = y_train[:-2000]

    # Define the neural network architecture
    net = {
        'layers': [
            {'type': 'input', 'params': {'size': (32, 32, 3)}},
            {'type': 'convolution', 'params': {'weights': 0.1 * np.random.randn(5, 5, 3, 16) / np.sqrt(5 * 5 * 3 / 2), 'biases': np.zeros((16,1))},
             'padding': [2, 2]},
            {'type': 'relu'},
            {'type': 'maxpooling'},
            {'type': 'convolution', 'params': {'weights': 0.1 * np.random.randn(5, 5, 16, 16) / np.sqrt(5 * 5 * 16 / 2), 'biases': np.zeros((16,1))},
             'padding': [2, 2]},
            {'type': 'relu'},
            {'type': 'fully_connected', 'params': {'weights': np.random.randn(10, 4096) / np.sqrt(4096 / 2), 'biases': np.zeros((10,1))}},
            {'type': 'softmaxloss'}
        ]
    }

    # Display layer sizes
    a, b = evaluate(net, x_train[:, :, :, :8], y_train[:8], True, True)

    # Define training options
    training_opts = {
        'learning_rate': 1e-3,
        'iterations': 5000,
        'batch_size': 16,
        'momentum': 0.95,
        'weight_decay': 0.001
    }

    # Train the neural network
    net, _ = training(net, x_train, y_train, x_val, y_val, training_opts, make_plots=True)

    # Save the trained model
    # You need to implement the saving mechanism based on your specific implementation
    # For example, using pickle or a custom save function

    # Evaluate on the test set
    pred = np.zeros(len(y_test))
    batch = training_opts['batch_size']
    for i in range(0, len(y_test), batch):
        idx = slice(i, min(i + batch, len(y_test)))
        y, _ = evaluate(net, x_test[:, :, :, idx], y_test[idx], evaluate_gradient=False)
        p = np.argmax(y[-2], axis=0)
        pred[idx] = p

    pred = pred + 1  # NOTE: cifar labels go from 1 to 10

    accuracy = np.mean(pred == y_test)
    print(f'Accuracy on the test set: {accuracy}')

def evaluate(net, inp, labels, evaluate_gradient=True, verbose=False):
    """
    Evaluate the neural network.

    Args:
    - net (dict): Neural network structure and parameters.
    - inp (numpy.ndarray): Input data.
    - labels (numpy.ndarray): Ground truth labels.
    - evaluate_gradient (bool): If True, evaluate gradients during backpropagation.
    - verbose (bool): If True, print layer information during evaluation.

    Returns:
    - res (list): List of intermediate results for each layer.
    - param_grads (dict): Dictionary containing parameter gradients if evaluate_gradient is True.

    Raises:
    - AssertionError: If the input dimensions do not match the expected dimensions for the input layer.
    """
    backprop = evaluate_gradient
    n_layers = len(net['layers'])

    input_size = inp.shape
    batch_size = input_size[-1]
    input_dims = input_size[:-1]

    res = [None] * n_layers

    for i in range(n_layers):
        layer = net['layers'][i]

        if i == 0:
            assert layer['type'] == 'input', 'The first layer must be an input layer'

        if layer['type'] == 'input':
            assert input_dims == layer['params']['size'], 'The input dimension is wrong'
            res[i] = inp
        elif layer['type'] == 'fully_connected':
            assert "params" in layer, 'Parameters for the fully connected layer are not specified'
            assert "weights" in layer["params"], 'The weights for the fully connected layer are not specified.'
            assert "biases" in layer["params"], 'The biases for the fully connected layer are not specified.'
            res[i] = fully_connected_forward(res[i-1], layer['params']['weights'], layer['params']['biases'])
            #if backprop:
            #    grad, param_grads[i] = fully_connected_backward(res[i-1], grad, layer['params']['weights'], layer['params']['biases'])
        elif layer['type'] == 'convolution':
            assert "params" in layer, 'Parameters for the convolution layer are not specified'
            assert "weights" in layer["params"], 'The weights for the convolution layer are not specified.'
            assert "biases" in layer["params"], 'The biases for the convolution layer are not specified.'

            padding = [0, 0]
            if 'padding' in layer:
                padding = layer['padding']
            res[i] = conv_with_padding_forward(res[i-1], layer['params']['weights'], layer['params']['biases'], padding)
            #if backprop:
            #    grad, param_grads[i] = conv_with_padding_backward(res[i-1], grad, layer['params']['weights'], layer['params']['biases'], padding)
        elif layer['type'] == 'maxpooling':
            res[i] = maxpooling_forward(res[i-1])
            #if backprop:
            #    grad = maxpooling_backward(res[i-1], grad)
        elif layer['type'] == 'relu':
            res[i] = relu_forward(res[i-1])
            #if backprop:
            #    grad = relu_backward(res[i-1], grad)
        elif layer['type'] == 'softmaxloss':
            res[i] = softmaxloss_forward(res[i-1], labels)
            #if backprop:
            #    grad = softmaxloss_backward(res[i-1], labels)
        else:
            raise ValueError(f'Unknown layer type {layer["type"]}')

        if verbose:
            print(f'Layer {i+1}, ({layer["type"]}) size ({res[i].shape})')

    assert res[-1].size == 1, 'The final output must be a single element, the loss.'

    param_grads = [None] * n_layers
    if backprop:
        grad = [None] * n_layers
        for i in range(n_layers-1, 0, -1):
            layer = net['layers'][i]

            if layer['type'] == 'input':
                raise ValueError('Do not backpropagate to the input')
            elif layer['type'] == 'fully_connected':
                grad_x, grad_w, grad_b = fully_connected_backward(res[i-1], grad[i+1]["grad"], layer['params']['weights'], layer['params']['biases'])
                grad[i] = {"grad": grad_x}
                param_grads[i] = {"weights": grad_w, "biases": grad_b}
            elif layer['type'] == 'convolution':
                padding = [0, 0]
                if 'padding' in layer:
                    padding = layer['padding']
                grad_x, grad_w, grad_b = conv_with_padding_backward(res[i-1], grad[i+1]["grad"], layer['params']['weights'], layer['params']['biases'], padding)
                grad[i] = {"grad": grad_x}
                param_grads[i] = {"weights": grad_w, "biases": grad_b.reshape((-1,1), order='F')}
            elif layer['type'] == 'maxpooling':
                grad[i] = {"grad": maxpooling_backward(res[i-1], grad[i+1]["grad"])}
            elif layer['type'] == 'relu':
                grad[i] = {"grad": relu_backward(res[i-1], grad[i+1]["grad"])}
            elif layer['type'] == 'softmaxloss':
                grad[i] = {"grad": softmaxloss_backward(res[i-1], labels)}

            if verbose:
                print(f'BP Layer {i+1}, ({layer["type"]}) size ({grad[i]["grad"].shape})')

    return res, param_grads

def mnist_starter():
    """
    Load and preprocess the MNIST dataset, define and train a convolutional neural network, and evaluate its performance.

    Returns:
    None
    """

    base_path = "/your/path/here/"
    mat_file_path = base_path + "train.mat"
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: File '{mat_file_path}' not found.")
        mat_data = None

    if mat_data is not None:
        x_train = mat_data['x_train']
        x_train = x_train.reshape((28,28,1,60000), order='F')
        y_train = mat_data['y_train']
        y_train[y_train==0] = 10
    mat_file_path = base_path + "test.mat"
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
    except FileNotFoundError:
        print(f"Error: File '{mat_file_path}' not found.")
        mat_data = None

    if mat_data is not None:
        x_test = mat_data['x_test']
        x_test = x_test.reshape((28,28,1,10000), order='F')
        y_test = mat_data['y_test']
        y_test[y_test == 0] = 10
    classes = [1,2,3,4,5,6,7,8,9,0]

    # Reshape training data
    x_val = x_train[:, :, :, -2000:]
    y_val = y_train[-2000:]
    x_train = x_train[:, :, :, 0:-2000]
    y_train = y_train[0:-2000]

    # Subtract mean intensity
    data_mean = np.mean(x_train)
    x_train -= data_mean
    x_val -= data_mean
    x_test -= data_mean

    # Define the neural network architecture
    net = {
        'layers': [
            {'type': 'input', 'params': {'size': (28, 28, 1)}},
            {'type': 'convolution', 'params': {'weights': np.random.randn(5, 5, 1, 16) / np.sqrt(5 * 5 * 3 / 2),
                                               'biases': np.zeros((16,1))}, 'padding': [2,2]},
            {'type': 'relu'},
            {'type': 'maxpooling'},
            {'type': 'convolution', 'params': {'weights': np.random.randn(5, 5, 16, 16) / np.sqrt(5 * 5 * 16 / 2),
                                               'biases': np.zeros((16,1))}, 'padding': [2,2]},
            {'type': 'relu'},
            {'type': 'maxpooling'},
            {'type': 'fully_connected', 'params': {'weights': np.random.randn(10, 784) / np.sqrt(784 / 2),
                                                    'biases': np.zeros((10,1))}},
            {'type': 'softmaxloss'}
        ]
    }

    # Print the layer sizes and make sure that all parameters have the correct sizes
    evaluate(net, x_train[:, :, :, 0:8], y_train[0:8], verbose=True, evaluate_gradient=False)

    # Training options
    training_opts = {
        'learning_rate': 1e-1,
        'iterations': 3000,
        'batch_size': 16,
        'momentum': 0.9,
        'weight_decay': 0.005
    }

    # Run the training
    net, _ = training(net, x_train, y_train, x_val, y_val, training_opts, make_plots=True)

    # Save the trained model
    np.save('./network_trained_with_momentum.npy', net)

    # Evaluate on the test set
    pred = np.zeros(len(y_test))
    batch = training_opts['batch_size']
    for i in range(0, len(y_test), batch):
        idx = slice(i, min(i + batch, len(y_test)))
        y, _ = evaluate(net, x_test[:, :, :, idx], y_test[idx], evaluate_gradient=False)
        p = np.argmax(y[-2], axis=0)
        pred[idx] = p

    pred = pred + 1  # Labels go from 1 to 10
    y_test = y_test.squeeze()

    accuracy = np.mean(pred == y_test)
    print(f'Accuracy on the test set: {accuracy:.4f}')

    # Plot some misclassified examples ... 

def training(net, x, labels, x_val, labels_val, opts, make_plots=False):
    """
    Train a neural network using gradient descent.

    Args:
    - net (Network): Neural network model.
    - x (numpy.ndarray): Training data of shape (height, width, channels, num_samples).
    - labels (numpy.ndarray): Training labels.
    - x_val (numpy.ndarray): Validation data of shape (height, width, channels, num_samples).
    - labels_val (numpy.ndarray): Validation labels.
    - opts (dict): Dictionary containing training options, including hyperparameters.

    Returns:
    - net (Network): Trained neural network model.
    - loss (numpy.ndarray): Array containing the total loss at each iteration.

    Raises:
    - ValueError: If the neural network is not provided.
    """
    if net is None:
        raise ValueError("Neural network 'net' must be provided.")

    loss = np.zeros(opts['iterations'])
    loss_weight_decay = np.zeros(opts['iterations'])
    loss_ma = np.zeros(opts['iterations'])
    accuracy = np.zeros(opts['iterations'])
    accuracy_ma = np.zeros(opts['iterations'])

    opts['moving_average'] = 0.995
    opts['print_interval'] = 100
    opts['validation_interval'] = 100
    opts['validation_its'] = 10

    sz = x.shape
    n_training = sz[3]
    val_it = [0]
    val_acc = [0]

    # Initialize momentum
    momentum = np.empty((len(net["layers"]), 1), dtype=dict)

    for it in range(opts['iterations']):
        # Extract the elements of the batch
        indices = np.random.choice(n_training, opts['batch_size'], replace=False)
        x_batch = x[:, :, :, indices]
        labels_batch = labels[indices]

        # Forward and backward pass of the network using the current batch
        z, grads = evaluate(net, x_batch, labels_batch, evaluate_gradient=True)
        loss[it] = z[-1]

        if np.isnan(loss[it]) or np.isinf(loss[it]):
            raise ValueError('Loss is NaN or inf. Decrease the learning rate or change the initialization.')

        # We have a fully connected layer before the softmax loss
        # The prediction is the index corresponding to the highest score
        pred = np.argmax(z[-2], axis=0)
        accuracy[it] = np.mean(labels_batch.reshape(-1) == pred + 1)

        if it < 20:
            loss_ma[it] = np.mean(loss[:(it+1)])
            accuracy_ma[it] = np.mean(accuracy[:(it+1)])
        else:
            loss_ma[it] = opts['moving_average'] * loss_ma[it - 1] + (1 - opts['moving_average']) * loss[it]
            accuracy_ma[it] = opts['moving_average'] * accuracy_ma[it - 1] + (1 - opts['moving_average']) * accuracy[it]

        # Gradient descent by looping over all parameters
        for i in range(1, len(net["layers"])):
            layer = net["layers"][i]

            # Does the layer have any parameters? In that case, we update
            if 'params' in layer.keys():
                params = layer["params"]

                if 'momentum' in opts and it == 0:
                    momentum[i] = {"weights" : np.zeros_like(net["layers"][i]["params"]["weights"]),
                                   "biases" : np.zeros_like(net["layers"][i]["params"]["biases"])}

                for s in params:
                    # Compute the weight decay loss
                    loss_weight_decay[it] += opts['weight_decay'] / 2 * np.sum(net["layers"][i]["params"][s] ** 2)

                    # Momentum and update
                    if 'momentum' in opts:
                        momentum[i][0][s] = opts['momentum'] * momentum[i][0][s] + (1 - opts['momentum']) * grads[i][s]
                        net["layers"][i]["params"][s] -= opts['learning_rate'] * (momentum[i][0][s] +
                                                          (opts['weight_decay'] * net["layers"][i]["params"][s]))
                    else:
                        # Run normal gradient descent if the momentum parameter is not specified
                        net["layers"][i]["params"][s] -= opts['learning_rate'] * (grads[i][s] +
                                                         opts['weight_decay'] * net["layers"][i]["params"][s])

        # Check the accuracy on the validation set
        if it % opts['validation_interval'] == 0:
            correct = np.zeros((opts['validation_its'], opts['batch_size']), dtype=bool)
            for k in range(opts['validation_its']):
                indices = np.random.choice(len(labels_val), opts['batch_size'], replace=False)
                x_batch_val = x_val[:, :, :, indices]
                labels_batch_val = labels_val[indices]

                z_val, _ = evaluate(net, x_batch_val, labels_batch_val, evaluate_gradient=False)
                pred_val = np.argmax(z_val[-2], axis=0)
                correct[k, :] = ((labels_batch_val.reshape(-1) - 1) == pred_val)

            val_it.append(it)
            val_acc.append(0.5 * val_acc[-1] + 0.5 * np.mean(correct))

        if it % opts['print_interval'] == 0:
            print(f'Iteration {it}:\n'
                  f'Classification loss: {loss_ma[it]}\n'
                  f'Weight decay loss: {loss_weight_decay[it - 1]}\n'
                  f'Total loss: {loss_ma[it] + loss_weight_decay[it - 1]}\n'
                  f'Training accuracy: {accuracy_ma[it]}\n'
                  f'Validation accuracy: {val_acc[-1]}\n')

    loss = loss_ma + loss_weight_decay

    if make_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.plot(loss_ma, label='Training loss', color='blue')
        ax1.set_title('Plot 1')
        ax1.legend()
        ax2.plot(accuracy_ma, label='Training accuracy', color='blue')
        ax2.plot(val_it, val_acc, label='Validation accuracy', color='red')
        ax2.set_title('Plot 2')
        ax2.legend()
        plt.tight_layout()
        plt.show()

    return net, loss

if __name__ == "__main__":
    mnist_starter()
    #cifar10_starter()
