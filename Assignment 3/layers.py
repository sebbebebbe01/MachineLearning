import numpy as np

def conv_with_padding_backward(x, dldy, w, b, padding):
    """
    Compute gradients for convolution with padding.

    Args:
    - x (numpy.ndarray): Input data of shape (height, width, channels).
    - dldy (numpy.ndarray): Gradient of the loss with respect to the output of the convolution.
    - w (numpy.ndarray): Convolutional filter/kernel.
    - b (numpy.ndarray): Bias term.
    - padding (tuple): Tuple containing two elements (p_i, p_j), representing padding in the vertical and horizontal directions.

    Returns:
    - dldx (numpy.ndarray): Gradient of the loss with respect to the input.
    - dldw (numpy.ndarray): Gradient of the loss with respect to the convolutional filter/kernel.
    - dldb (numpy.ndarray): Gradient of the loss with respect to the bias term.

    Raises:
    - AssertionError: If the length of the padding vector is not equal to 2.
    """
    assert len(padding) == 2, 'There must be two elements in the padding vector.'
    p_i, p_j = padding
    sz_t = x.shape
    sz = (sz_t[0] + 2*p_i, sz_t[1] + 2*p_j, sz_t[2], sz_t[3])

    x_new = np.zeros(sz)
    x_new[p_i:sz[0]-p_i, p_j:sz[1]-p_j, :] = x
    dldx, dldw, dldb = convolution_backward(x_new, dldy, w, b)

    dldx = dldx[p_i:sz[0]-p_i, p_j:sz[1]-p_j, :, :]
    
    return dldx, dldw, dldb

def conv_with_padding_forward(x, w, b, padding):
    """
    Perform convolution with padding on input data.

    Args:
    - x (numpy.ndarray): Input data of shape (height, width, channels).
    - w (numpy.ndarray): Convolutional filter/kernel.
    - b (numpy.ndarray): Bias term.
    - padding (tuple): Tuple containing two elements (p_i, p_j), representing padding in the vertical and horizontal directions.

    Returns:
    - y (numpy.ndarray): Output after convolution with padding.

    Raises:
    - AssertionError: If the length of the padding vector is not equal to 2.
    """
    assert len(padding) == 2, 'There must be two elements in the padding vector.'
    p_i, p_j = padding
    sz_t = x.shape
    sz = (sz_t[0] + 2*p_i, sz_t[1] + 2*p_j, sz_t[2], sz_t[3])

    x_new = np.zeros(sz)
    x_new[p_i:sz[0]-p_i, p_j:sz[1]-p_j, :, :] = x
    y = convolution_forward(x_new, w, b)

    return y

def convolution_backward(x, dldy, w, b):
    """
    Compute gradients for the backward pass of a convolution operation.

    Args:
    - x (numpy.ndarray): Input data of shape (height, width, channels, batch_size).
    - dldy (numpy.ndarray): Gradient of the loss with respect to the output of the convolution.
    - w (numpy.ndarray): Convolutional filter/kernel.
    - b (numpy.ndarray): Bias term.

    Returns:
    - dldx (numpy.ndarray): Gradient of the loss with respect to the input.
    - dldw (numpy.ndarray): Gradient of the loss with respect to the convolutional filter/kernel.
    - dldb (numpy.ndarray): Gradient of the loss with respect to the bias term.

    Raises:
    - ValueError: If the dimensions of the input x, filter w, or bias b are incompatible.
    """
    sz = x.shape
    f_i, f_j, f_c, f_o = w.shape

    batch_size = sz[3]
    patches = (sz[0] - f_i + 1) * (sz[1] - f_j + 1)

    x_n = np.zeros((patches * batch_size, f_i * f_j * f_c))

    idx = 0
    for j in range(sz[1] - f_j + 1):
        for i in range(sz[0] - f_i + 1):
            x_n[idx:idx + batch_size, :] = np.reshape(x[i:i + f_i, j:j + f_j, :, :], [f_i * f_j * f_c, batch_size], order='F').T
            idx += batch_size

    dldF = np.dot(x_n.T, np.reshape(np.transpose(dldy, (3, 0, 1, 2)), [patches * batch_size, f_o], order='F'))
    dldw = np.reshape(dldF, [f_i, f_j, f_c, f_o], order='F')

    flipped_w = np.transpose(w[::-1, ::-1, :, :], [0, 1, 3, 2])
    dldx = conv_with_padding_forward(dldy, flipped_w, np.zeros([f_c, 1]), [f_i - 1, f_j - 1])

    dldb = np.sum(np.sum(np.sum(dldy, axis=1), axis=0), axis=1)

    return dldx, dldw, dldb

def convolution_forward(x, w, b):
    """
    Perform forward pass of a convolution operation.

    Args:
    - x (numpy.ndarray): Input data of shape (height, width, channels, batch_size).
    - w (numpy.ndarray): Convolutional filter/kernel of shape (filter_height, filter_width, input_channels, output_channels).
    - b (numpy.ndarray): Bias term of shape (output_channels,).

    Returns:
    - y (numpy.ndarray): Output of the convolution operation, of shape (output_height, output_width, output_channels, batch_size).

    Raises:
    - AssertionError: If the number of filter channels does not match the input channels or the number of bias elements.
    """
    sz = x.shape
    f_i, f_j, f_c, f_o = w.shape
    
    assert sz[2] == f_c, 'Filter channels did not match input channels'
    assert len(b) == f_o, 'Expected the same number of filters as bias elements'
    
    batch_size = sz[-1]
    patches = (sz[0] - f_i + 1) * (sz[1] - f_j + 1)
    
    F = np.reshape(w, [f_i * f_j * f_c, f_o], order='F')
    X = np.zeros((patches * batch_size, f_i * f_j * f_c))
    
    idx = 0
    for j in range(0, sz[1] - f_j + 1):
        for i in range(0, sz[0] - f_i + 1):
            X[idx:idx+batch_size, :] = np.reshape(x[i:i+f_i, j:j+f_j, :, :], [f_i * f_j * f_c, batch_size], order='F').T
            idx += batch_size
    
    Y = np.dot(X, F)
    y = np.reshape(Y, [batch_size, sz[0] - f_i + 1, sz[1] - f_j + 1, f_o], order='F')
    y = np.transpose(y, [1, 2, 3, 0])
    
    y = np.add(y, np.reshape(b, [1, 1, len(b), 1]))

    return y

def fully_connected_backward(X, dldZ, W, b):
    """
    Compute the gradients for the fully connected layer.

    Args:
    - X (numpy.ndarray): The input variable. The size might vary, but the last dimension
      tells you which element in the batch it is.
    - dldZ (numpy.ndarray): The partial derivatives of the loss with respect to the
      output variable Z. The size of dldZ is the same as for Z as computed in the forward pass.
    - W (numpy.ndarray): The weight matrix.
    - b (numpy.ndarray): The bias vector.

    Returns:
    - dldX (numpy.ndarray): Gradient backpropagated to X.
    - dldW (numpy.ndarray): Gradient backpropagated to W.
    - dldb (numpy.ndarray): Gradient backpropagated to b.

    All gradients should have the same size as the variable. That is,
    dldX and X should have the same size, dldW and W the same size, and dldb
    and b the same size.

    Raises:
    - AssertionError: If the number of columns in W does not match the number of features in X,
      or if the number of rows in W is not equal to the number of elements in b.
    """
    sz = X.shape
    batch = sz[-1]
    features = np.prod(sz[:-1])

    # Reshape the input vector so that all features for a single batch
    # element are in the columns. X is now as defined in the assignment.
    X = np.reshape(X, (features, batch), order='F')

    # print(dldZ.shape)
    # print(dldZ)

    assert W.shape[1] == features, f"Expected {features} columns in the weights matrix, got {W.shape[1]}"
    assert W.shape[0] == len(b), "Expected as many rows in W as elements in b"

    # Implement it here.
    # Note that dldX should have the same size as X, so use reshape before returning the output
    # as suggested.
    dldX = (W.T @ dldZ).reshape(sz, order='F')
    dldW = dldZ @ X.T
    dldb = dldZ @ np.ones((batch,1))

    return dldX, dldW, dldb

def fully_connected_forward(X, W, b):
    """
    Perform the forward pass for a fully connected layer.

    Args:
    - X (numpy.ndarray): Input variable. The size might vary, but the last dimension
                         indicates which element in the batch it is.
    - W (numpy.ndarray): Weight matrix.
    - b (numpy.ndarray): Bias vector.

    Returns:
    - Z (numpy.ndarray): Result of the forward pass.

    Raises:
    - AssertionError: If the number of columns in W does not match the number of features.
                      If the number of rows in W does not match the number of elements in b.
    """
    # Get input dimensions
    sz = X.shape
    batch = sz[-1]
    features = np.prod(sz[:-1])

    # Reshape input for matrix multiplication
    X = X.reshape((features, batch), order='F')

    # Check dimensions
    assert W.shape[1] == features, f'Expected {features} columns in the weights matrix, got {W.shape[1]}'
    assert W.shape[0] == len(b), 'Expected as many rows in W as elements in b'

    # Perform the forward pass
    Z = W@X + b # The addition of b (size nx1) to W@X (size nxN) works thanks to broadcasting

    return Z

def maxpooling_backward(x, dldy):
    """
    Compute gradients for max-pooling operation in the backward pass.

    Args:
    - x (numpy.ndarray): Input data of shape (height, width, channels, batch_size).
    - dldy (numpy.ndarray): Gradient of the loss with respect to the output of the max-pooling operation.

    Returns:
    - dldx (numpy.ndarray): Gradient of the loss with respect to the input.

    Note:
    - Assumes 2x2 max-pooling with stride 2.

    Raises:
    - ValueError: If the input data shape is not compatible.
    """
    if x.shape[0] % 2 != 0 or x.shape[1] % 2 != 0:
        raise ValueError("Input data shape must be divisible by 2 for 2x2 max-pooling.")

    y = np.stack([x[0::2, 0::2, :, :], x[0::2, 1::2, :, :], x[1::2, 0::2, :, :], x[1::2, 1::2, :, :]], axis=4)
    
    idx = np.argmax(y, axis=4)
    dldx = np.zeros_like(x)
    dldx[0::2, 0::2, :, :] = (idx == 0) * dldy
    dldx[0::2, 1::2, :, :] = (idx == 1) * dldy
    dldx[1::2, 0::2, :, :] = (idx == 2) * dldy
    dldx[1::2, 1::2, :, :] = (idx == 3) * dldy
    
    return dldx

def maxpooling_forward(x):
    """
    Perform max pooling on the input.

    Args:
    - x (numpy.ndarray): Input data of shape (height, width, channels).

    Returns:
    - y (numpy.ndarray): Output data after max pooling.

    Raises:
    - AssertionError: If the first and second dimensions of the input are not divisible by two.
    """
    assert (x.shape[0] % 2 == 0) and (x.shape[1] % 2 == 0), 'For maxpooling, the first and second dimension must be divisible by two.'

    y = np.stack([x[0::2, 0::2, :, :], x[0::2, 1::2, :, :], x[1::2, 0::2, :, :], x[1::2, 1::2, :, :]], axis=4)
    y = np.max(y, axis=4)

    return y

def relu_backward(X, dldZ):
    """
    Compute the gradient of the loss with respect to the input of the ReLU activation function.

    Args:
    - X (numpy.ndarray): Input data.
    - dldZ (numpy.ndarray): Gradient of the loss with respect to the output of the ReLU.

    Returns:
    - dldX (numpy.ndarray): Gradient of the loss with respect to the input of the ReLU.

    """
    
    # Implement here
    dldX = np.heaviside(X, 1/2) * dldZ
    return dldX

def relu_forward(X):
    """
    Compute the forward pass of the Rectified Linear Unit (ReLU) activation function.

    Args:
    - X (numpy.ndarray): Input data.

    Returns:
    - Z (numpy.ndarray): Output data after applying ReLU activation.
    """
    
    # Implement here
    Z = np.maximum(X,0)
    return Z

def softmaxloss_backward(x, labels):
    """
    Compute the partial derivative of the softmax loss with respect to the input.

    Args:
    - x (numpy.ndarray): Features. It should be reshaped as for the fully connected layer.
    - labels (numpy.ndarray): Vector with correct labels. For instance, if we have a batch of two where the first example is class 4 and the second example is class 7, labels is [4, 7].

    Returns:
    - dldx (numpy.ndarray): Partial derivative of the loss with respect to x. Average over the batch elements in the forward pass.

    Raises:
    - ValueError: If the size of x and labels are not compatible for matrix multiplication.
    """
    labels = np.ravel(labels)
    sz = x.shape
    batch = sz[-1]
    features = np.prod(sz[:-1])

    # Suitable for matrix multiplication
    x = np.reshape(x, [features, batch], order='F')

    # For numerical stability. Convince yourself that the result is the same.
    x = x - np.min(x, axis=0, keepdims=True)
    
    # labels -= 1 # NO!! The labels are already shifted down by one

    # Implement here
    exp = np.exp(x)
    y = exp / np.sum(exp, axis=0)
    y[labels, np.arange(batch)] -= 1 # Remove one from the indices where c==i
    dldx = 1/batch * y

    return dldx

def softmaxloss_forward(x, labels):
    """
    Compute the softmax loss for a batch of examples.

    Args:
    - x (numpy.ndarray): Input features. Reshaped as for the fully connected layer.
    - labels (numpy.ndarray or list): Vector with correct labels for each example in the batch.

    Returns:
    - L (float): Computed loss. The average loss over the batch elements.

    Raises:
    - AssertionError: If the number of labels does not match the batch size.
    """
    labels = np.array(labels).flatten()
    sz = x.shape
    batch = sz[-1]
    features = np.prod(sz[:-1])

    assert batch == len(labels), 'Wrong number of labels given'
    
    # Reshape x as for the fully connected layer
    x = np.reshape(x, (features, batch), order='F')
    
    # For numerical reasons, subtract the minimum value along each column
    x = x - np.min(x, axis=0)
    
    # Implement here
    x_c = x[labels, np.arange(batch)] # An array with all x_c in the batch
    L = np.mean(-x_c + np.log(np.sum(np.exp(x),axis=0)))
    return L

