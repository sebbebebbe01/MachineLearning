import numpy as np

from layers import (fully_connected_forward, fully_connected_backward, relu_forward, 
                    relu_backward, softmaxloss_backward, softmaxloss_forward,
                    convolution_forward)
from starters import evaluate
import time


# NOTE: np.allclose with reasonable thresholds could also be used
def test_equal(a, b, epsilon=1e-7, msg=""):
    """
    Compare two arrays for equality up to a specified epsilon.

    Args:
    - a (numpy.ndarray): First array for comparison.
    - b (numpy.ndarray): Second array for comparison.
    - epsilon (float, optional): Tolerance for considering values equal. Default is 1e-7.
    - msg (str, optional): Custom error message to display in case of test failure. Default is an empty string.

    Raises:
    - AssertionError: If the sizes of the input arrays do not match or if the arrays are not equal within the specified tolerance.
    """
    if a.shape != b.shape:
        raise AssertionError('Sizes do not match')

    a_ = a.copy()
    b_ = b.copy()
    a = a.flatten()
    b = b.flatten()

    if np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-7)) > epsilon:
        print('######### FAILED TEST ##################')
        print(f'Expected: {b_}, got: {a_}')
        raise AssertionError(msg)

def test_fully_connected():
    """
    Perform various tests to verify the correctness of the fully connected layer implementation.

    This function checks forward passes and performs exhaustive gradient checking.

    Raises:
    - AssertionError: If any of the tests fail.
    """
    np.set_printoptions(precision=5, suppress=True)

    # Add necessary imports here
    # from your_module import fully_connected_forward, fully_connected_backward, test_equal, test_gradients

    # Tests for forward passes
    # Test 1
    x = np.zeros((2, 1))
    x[:, 0] = [1, 2]
    A = np.array([[-1, 3], [7, -3]])
    b = np.zeros((2, 1))
    res = np.array([5, 1]).reshape(-1,1)
    test_equal(fully_connected_forward(x, A, b), res, 1e-5, 'Forward pass with a batch of one is incorrect')

    # Add more forward pass tests

    # Tests for gradients
    for _ in range(5):
        # Test gradient calculations
        x = np.random.randn(21, 4)
        A = np.random.randn(3, 21)
        b = np.random.randn(3, 1)
        y = fully_connected_forward(x, A, b)
        dldx, dldA, dldb = fully_connected_backward(x, y, A, b)
        test_gradients(lambda x: 0.5 * np.sum((fully_connected_forward(x, A, b)**2)),
                       dldx, x, 1e-5, 30)
        test_gradients(lambda A: 0.5 * np.sum((fully_connected_forward(x, A, b)**2)),
                       dldA, A, 1e-5, 30)
        test_gradients(lambda b: 0.5 * np.sum((fully_connected_forward(x, A, b)**2)),
                       dldb, b, 1e-5, 5)

    for _ in range(5):
        x = np.random.randn(5, 3, 4)
        A = np.random.randn(3, 15)
        b = np.random.randn(3, 1)
        y = fully_connected_forward(x, A, b)
        dldx, dldA, dldb = fully_connected_backward(x, y, A, b)
        test_gradients(lambda x: 0.5 * np.sum((fully_connected_forward(x, A, b)**2)),
                       dldx, x, 1e-5, 30)
        test_gradients(lambda A: 0.5 * np.sum((fully_connected_forward(x, A, b)**2)),
                       dldA, A, 1e-5, 30)
        test_gradients(lambda b: 0.5 * np.sum((fully_connected_forward(x, A, b)**2)),
                       dldb, b, 1e-5, 5)

    print('Everything passed! Your implementation seems correct.')

def test_gradient_whole_net():
    """
    Test the gradient using the evaluate function.

    Creates a simple neural network with an input layer, a fully connected layer,
    a ReLU activation layer, and a softmax loss layer. Then, it evaluates the
    gradients using the test_gradients function.

    Returns:
    None

    Raises:
    AssertionError: If any gradient test fails.
    """
    np.random.seed(0)  # For reproducibility
    net = {'layers': []}

    net['layers'].append({'type': 'input', 'params': {'size': (3, 2)}})
    net['layers'].append({'type': 'fully_connected', 'params': {'weights': np.random.randn(3, 6), 'biases': np.random.randn(3, 1)}})
    net['layers'].append({'type': 'relu'})
    net['layers'].append({'type': 'softmaxloss'})

    for _ in range(3):
        x = np.random.randn(3, 2, 2)

        labels = np.random.randint(3, size=(2, 1))
        y, grads = evaluate(net, x, labels, evaluate_gradient=True)

        test_gradients(lambda y: change_one_param(y, net, x, labels, 1, 'biases'), grads[1]['biases'], net['layers'][1]['params']['biases'], 1e-5, 5)
        test_gradients(lambda y: change_one_param(y, net, x, labels, 1, 'weights'), grads[1]['weights'], net['layers'][1]['params']['weights'], 1e-5, 30)
        # print('Loss', y[-1])

    print('Everything passed!')

def change_one_param(param, net, x, labels, layer_id, name):
    """
    Change one parameter in the network, evaluate the result, and return the loss.

    Args:
    - param: New parameter value.
    - net: Neural network dictionary.
    - x: Input data.
    - labels: Ground truth labels.
    - layer_id: Index of the layer to change.
    - name: Name of the parameter to change.

    Returns:
    float: Loss value.

    Raises:
    None
    """
    x0 = net['layers'][layer_id]['params'][name]
    net['layers'][layer_id]['params'][name] = param
    res, grads = evaluate(net, x, labels, evaluate_gradient=True)
    y = res[-1]
    net['layers'][layer_id]['params'][name] = x0
    if isinstance(y, np.float64):
        return y
    x0 = net['layers'][layer_id]['params'][name]
    net['layers'][layer_id]['params'][name] = param
    res = evaluate(net, x, labels)
    y = res[-1]
    net['layers'][layer_id]['params'][name] = x0

def test_gradients(f, g, x0, epsilon, n):
    """
    Test gradients of a function using finite differences.

    Args:
    - f (function): Function to be tested.
    - g (numpy.ndarray): Gradient of the function.
    - x0 (numpy.ndarray): Initial point for testing.
    - epsilon (float): Small perturbation for finite differences.
    - n (int): Number of random indices to test.

    Raises:
    - AssertionError: If the gradient has a different dimensionality than expected.
    - AssertionError: If the relative difference between finite differences and backpropagation is unacceptably large.
    """
    reference = g  # g(x0)

    assert np.array_equal(x0.shape, g.shape), 'The gradient has a different dimensionality than expected. Make sure that, e.g., x and dLdx have the exact same shape.'

    print('Gradient testing')
    print('       idx  {:13s}{:13s}{:13s}'.format('Fin diff', 'Backprop', 'Rel. diff'))
    for k in range(n):
        idx = np.random.randint(np.prod(x0.shape))
        pos = x0.copy()
        pos.flat[idx] = pos.flat[idx] + epsilon
        neg = x0.copy()
        neg.flat[idx] = neg.flat[idx] - epsilon
        d = (f(pos) - f(neg)) / (2 * epsilon)
        r = reference.flat[idx]
        rel = abs(d - r) / (abs(d) + abs(r) + 1e-8)
        print('It {:2d} {:3d}: {:13.5e} {:13.5e} {:13.4e}'.format(k, idx, d, r, rel))

        if rel > 1e-5:
            raise AssertionError('Unacceptably large gradient error.')

def test_relu():
    """
    Test the ReLU activation function and its gradients.

    Tests include forward pass with a batch of one and gradients computation.

    Raises:
    - AssertionError: If any test fails.
    """
    x = np.array([[2, 4, -5, 6, 7, -1],
                  [-4, 0, 0, 3, 4, -5]])
    res = np.array([[2, 4, 0, 6, 7, 0],
                    [0, 0, 0, 3, 4, 0]])
    test_equal(relu_forward(x), res, 1e-5, 'Forward pass with a batch of one is incorrect')

    # Gradients
    for _ in range(5):
        x = np.random.randn(21, 34, 4)
        y = relu_forward(x)
        dldx = relu_backward(x, y)
        test_gradients(lambda x: 0.5 * np.sum(np.square(relu_forward(x))), dldx, x, 1e-5, 30)

    # A few special values of the gradient. Verify with pen and paper.
    test_equal(relu_backward(np.ones((1,1)), np.ones((1,1))), np.ones((1,1)), 1e-5, 'The relu_backward function is incorrect.')
    test_equal(relu_backward(-np.ones((1,1)), np.ones((1,1))), np.zeros((1,1)), 1e-5, 'The relu_backward function is incorrect.')

    print('Everything passed! Your implementation seems correct.')

def test_softmaxloss():
    """
    Test the softmax loss implementation.

    Raises:
    - AssertionError: If the forward pass or gradients are incorrect.
    """
    x = np.array([[1, 2, -1],
                  [2, 4, -2]])
    labels = np.array([0, 1, 0])
    res = 1/3 * ((-1 + np.log(np.exp(1) + np.exp(2))) +
                 (-4 + np.log(np.exp(2) + np.exp(4))) +
                 (1 + np.log(np.exp(-1) + np.exp(-2))))
    test_equal(softmaxloss_forward(x, labels), res,
               1e-5, 'The forward pass is incorrect.')

    # gradients
    for _ in range(5):
        x = np.random.randn(5, 4)
        labels = np.random.randint(0, 5, size=(1, 4))
        y = softmaxloss_forward(x, labels)
        dldx = softmaxloss_backward(x, labels)
        # no need to square this time since we have scalar output
        test_gradients(lambda x: softmaxloss_forward(x, labels), dldx, x, 1e-5, 30)

    print('Everything passed! Your implementation seems correct.')

def tests():
    """
    Run a series of tests for convolution, max pooling, and their combinations.
    """
    np.random.seed(42)  # Set a seed for reproducibility

    # Add the necessary imports or replace the functions with the corresponding Python implementations

    # Identity convolution
    x = np.random.randn(5, 5, 3, 2)
    w = np.zeros((1, 1, 3, 3))
    w[0, 0, 0, 0] = 1
    w[0, 0, 1, 1] = 1
    w[0, 0, 2, 2] = 1
    b = np.zeros((3, 1))

    test_equal(convolution_forward(x, w, b), x, 1e-5, 'Identity convolution')

    # Impulse
    x = np.zeros((5, 5, 1, 1))
    x[2, 2, 0, 0] = 1
    w = np.ones((5, 5, 1, 1))
    b = np.zeros(1)
    test_equal(convolution_forward(x, w, b), np.array([1]), 1e-5, 'Impulse')

    # Other tests ...

    # Note: You need to implement the corresponding Python functions (convolution_forward, etc.) and
    # any additional helper functions they rely on. Replace the placeholders accordingly.

    print('Everything passed!')

if __name__ == "__main__":
    start = time.time()
    # test_fully_connected()
    # test_relu()
    # test_softmaxloss()
    test_gradient_whole_net()

    end = time.time()

    print('Elapsed time: ', str(end-start))