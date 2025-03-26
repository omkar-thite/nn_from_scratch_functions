import numpy as np
import matplotlib.pyplot as plt

# Initializations
def initialize_parameters_he(layers_dims):
    """
    Args:
    layer_dims(list) : Size of each layer (including input layer)
    
    Returns:
    parameters (dict): Randomly initialized parameters Ws and bs with He initialization .

    Notes:
    Ws are initialized randomly whereas biases are initialized to zero.
    """
    
    parameters = {}
    for i in range(1, len(layers_dims)):
        parameters['W'+str(i)] = np.random.randn(layers_dims[i], layers_dims[i-1]) * np.sqrt(2.0/layers_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layers_dims[i], 1))
    return parameters   


def initialize_parameters_LeCun(layers_dims: list, verbose: bool = False) -> dict:
    ''' 
    Initializes parameters with LeCun initialization (for sigmoid/tanh activations)

    Args:
    layer_dims (list): Contains number of neurons per layer in order, including those of input layer.
    verbose (bool): 

    Returns:
    parameters (dict): Initialized parameters according to LeCun initialization 

    Notes:
    Weights are initialized with Lecun Initialization, biases are initialized to zero.
    '''
    
    L = len(layers_dims)    
    parameters = {}
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))
        
    if verbose:
        print(f'Initialized parameters:')
        for key in parameters:
            print(f'{key} shape: {parameters[key].shape}')
        
    return parameters


def crossentropy_cost(AL: np.ndarray, Y: np.ndarray, epsilon: float=1e-8):
    '''
    Calculates cross entropy cost.

    Args:
    AL (ndarray): Activations of last layer (L) for all examples
    Y (ndarray): Ground labels for all examples
    epsilon (float): Small value to prevent log(0) errors (default: 1e-8) 

    Returns: 
    J (float): Crossentropy cost of forward pass for all examples.

    Raises:
    ValueError: If AL and Y shapes do not match.
    '''
     # Fast assertion for shape matching
    if AL.shape != Y.shape:
        raise ValueError(f"Dimension mismatch: AL {AL.shape} != Y {Y.shape}")

    # Clip AL to avoid log(0) or log(1) cases
    AL = np.clip(AL, epsilon, 1 - epsilon)
    
    J = - np.mean((Y * np.log(AL)) + (1 - Y) * np.log(1-AL)) 
    return float(J)

def cost_grad(AL: np.ndarray, Y: np.ndarray, epsilon: float=1e-8) -> np.ndarray:
    '''
    Computes gradient of cross-entropy function.

    Args:
    AL (ndarray) (1, number_of_examples): probability vector of predictions
    Y (ndarray) (1, number_of_examples): True labels 
    epsilon (float): small constant to avoid division by zero
    
    Returns:
    dAL (ndarray) (1, number_of_examples): gradient of the cost with respect to AL for all examples
    '''
    return -(np.divide(Y, AL+epsilon) - np.divide(1 - Y, 1 - AL+epsilon))


def relu_prime(Z: np.ndarray):
    '''
    Computes gradient of relu function.

    Args:
    Z (ndarray): Linear ouput of forward pass

    Returns
    np.ndarray: Gradient of ReLU, where values are 1 for Z >= 0 and 0 otherwise.
    '''
    return np.where(Z >= 0, 1, 0)
  

def relu(Z: np.ndarray) -> np.ndarray:
    '''
    Applies relu on Z

    Args:
    Z (ndarray): Linear ouput of forward pass.

    Returns:
    np.ndarray : ReLu activation
    '''
    return np.maximum(0, Z)

def sigmoid_prime(Z: np.ndarray) -> np.ndarray:
    '''
    Computes gradient of sigmoid function.

    Args:
    Z (ndarray): Linear ouput of forward pass.

    Returns: 
    np.ndarray: Gradient of sigmoid over all examples
    '''
    a = sigmoid(Z)
    return a * (1 - a)

def sigmoid(Z: np.ndarray) -> np.ndarray:
    '''
    Computes sigmoid function

    Args:
    Z (ndarray): Linear ouput of forward pass.
    
    Returns:
    np.ndarray : sigmoid activation
    '''
    # Clip extreme values to prevent overflow
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))


def forward_linear(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Implements forward linear pass Z = A_prev.W + b

    Args:
    A_prev (ndarray): Activations from previous layer
    W (ndarray): Weights of current layer
    b (ndarray): Biases of current layer

    Returns:
    Z (ndarray): Output of forward pass through layer
    '''
    assert A_prev.shape[0] == W.shape[1], "Dimension mismatch: A_prev and W"
    assert b.shape[0] == W.shape[0] and b.shape[1] == 1, "Dimension mismatch: b"
    
    return np.dot(W, A_prev) + b
    

def forward_activation(Z: np.ndarray, activation: str='linear') -> np.ndarray:
    '''
    Implements activation part of forward pass, A = activation(Z)

    Args: 
    Z (ndarray): Output of forward pass through current layer
    activation (str): Activation to apply on Z, one of 'sigmoid' or 'relu', 'linear' (default)

    Returns:
    np.ndarray: Activaitons of current layer
    '''  
    if activation=='relu':
        return relu(Z)        
    elif activation=='sigmoid':
        return sigmoid(Z)        
    else:
        return Z


def forward_propagation(X: np.ndarray, parameters: dict) -> tuple: 
    '''
    Performs forward propagation through a neural network.

    Args:
    X (ndarray): Input data
    paramaters (dict): Dictionary containing initialized network parameters:

    Returns:
    tuple: (AL, caches)
        - AL (ndarray) (1, number_of_examples): Output of the final layer (sigmoid activation)
        - caches (dict): Dictionary of dictionary: containing cached values needed for backpropagation
    '''
    
    def propagate_forward(A_prev, W, b, activation='relu'):
        '''
        Forward pass through layer

        Args:
        W (ndarray): Weights of layer
        b (ndarray): Biases of layer
        activation (str): Activation to apply on linear output, one of 'sigmoid' or 'relu'

        Returns:
        (ndarray): Output of forward pass through layer
        cache : Cache of current layer
        '''
         # Linear forward -> ReLU activation
        Z = forward_linear(A_prev, W, b)
        A = forward_activation(Z, activation=activation)
        cache = {'A_prev': A_prev, 'Z': Z, 'W': W, 'b':b, 'activation':activation} 

        return A, cache
          
    caches= {}
    A_prev = X
    L = len(parameters)//2
    
    # Process layers 1 to L-1 with ReLU activation
    for l in range(1, L):
        
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        
        A, cache = propagate_forward(A_prev, W, b, 'relu')
        
        # Store cache for current layer
        caches[f'layer{l}'] = cache
        
        A_prev = A
    
    #Process output layer L with sigmoid activation
    W = parameters[f'W{L}']
    b = parameters[f'b{L}']

    AL, cache =  propagate_forward(A_prev, W, b, 'sigmoid')
        
    # Store cache for current layer
    caches[f'layer{L}'] = cache
    
    return (AL, caches)


def backprop_layer(dA, cache):
    '''
    Implements backward pass for one layer

    Args:
    dA (ndarray): Gradient of the cost with respect to the activation of the current layer
    cache (dict): dictionary received from forward pass of current layer

    Returns:
    tuple: (dA_prev, dW, db)
        - dA_prev (ndarray): Gradient of the cost with respect to the activation of the previous layer
        - dW (ndarray): Gradient of the cost with respect to the weights of the current layer
        - db (ndarray): Gradient of the cost with respect to the bias of the current layer
    ''' 
    def d_Z(dA: np.ndarray, Z: np.ndarray, activation: str=None) -> np.ndarray:
        '''
        Computes gradient of cost wrt Z.
    
        Args:
        dA (ndarray) (1, number_of_examples): gradient vector of A
        Z (ndarray): Linear output of forward pass
        activation (str): Activation function, one of 'relu' or 'sigmoid'.
    
        Returns: 
        (ndarray): gradient of cost wrt Z
        '''
        if activation == 'relu':
            return dA * relu_prime(Z)
        elif activation == 'sigmoid':
            return dA * sigmoid_prime(Z)
        else:
            return dA #linear activation
    
    # Unpack cache values
    A_prev, Z, W, b, activation = cache['A_prev'], cache['Z'], cache['W'], cache['b'], cache['activation']
    m = dA.shape[1]   # Number of examples

    # Compute gradient of Z based on activation function
    dZ = d_Z(dA, Z, activation)

    # Compute gradients for parameters and previous layer
    dA_prev =  np.dot(W.T, dZ) 
    dW = (1/m) * np.dot(dZ, A_prev.T) 
    db = (1/m) * np.sum(dZ,axis=1, keepdims=True) 
    
    return (dA_prev, dW, db)

def backward_propagation(AL, Y, caches):
    '''
    Complete backward propagation through all layers
    
    Args:
    AL (ndarray): Output of the forward propagation
    Y (ndarray): True labels
    caches (dict): Dictionary of cached values from forward pass for current layer
    
    Returns:
    dict: Gradients for all parameters
    '''
    m = Y.shape[1]
    grads = {}
    L = len(caches)
    
    # Initialize backward propagation with output layer
    dAL = cost_grad(AL, Y)
    grads[f'dA{L}'] = dAL
    
    # Process all layers in reverse
    dA = dAL
    for l in reversed(range(1, L + 1)):

        cache = caches[f'layer{l}']
        dA_prev, dW, db = backprop_layer(dA, cache)
        
        # Store gradients
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
        
        if l > 1:  # Only store intermediate dA if not the input layer
            grads[f'dA{l-1}'] = dA_prev

        # Update for next iteration
        dA = dA_prev
        
    return grads


def L2_backpropagation(AL: np.ndarray, Y: np.ndarray, caches: dict, lambd: float=0.0) -> dict: 
    """
    Implementation of backpropagation with L2 regularization.
    
    Args:
    AL (ndarray) (1, number of examples): output of the forward propagation
    Y (ndarray) (1, number of examples): true label vector
    caches (dict): dictionary of cached values from forward pass
    lambd (scalar): regularization hyperparameter
    
    Returns:
    grads (dict): gradients

    Raises:
    TypeError: if caches is not a dictionary
    ValueError: if lambd parameter is negative
    """
    # Input validation
    if not isinstance(caches, dict):
        raise TypeError("caches must be a dictionary.")
    if lambd < 0:
        raise ValueError("lambd must be non-negative")
        
    L = len(caches)
    m = Y.shape[1]

    # Get gradients from regular backprop
    grads = backward_propagation(AL, Y, caches)

    # Add L2 regularization term to the gradients
    for l in range(1, L+1):
        cache = caches[f'layer{l}']   
        grads[f'dW{l}'] += (lambd/m) * cache['W']
        
    return grads

def L2_cost(AL: np.ndarray, Y: np.ndarray, parameters: dict,  lambd: float=0.0) -> float:
    """
    Computes the cost with L2 regularization.
    
    Args:
    AL (ndarray) (1, number of examples): output of the forward propagation
    Y (ndarray) (1, number of examples): true label vector
    parameters (dict): dictionary containing model parameters 
    lambd (scalar): regularization hyperparameter, default=0.0.
    
    Returns:
    total_cost (float): cross-entropy cost with L2 regularization.

    Raises:
    TypeError: if parameters is not a dictionary
    ValueError: if lambd parameter is negative
    """
    # Input validation
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary")
    if lambd < 0:
        raise ValueError("lambd must be non-negative")
        
    m = Y.shape[1]  # number of examples

     # Compute the number of layers 
    L = len(parameters)//2

    # Compute cross-entropy cost
    cross_entropy_cost = crossentropy_cost(AL, Y)

    # Compute L2 regularization cost
    L2_regularization_cost = 0
    for l in range(1, L+1):
        L2_regularization_cost += np.sum(np.square(parameters[f'W{l}']))
    
    L2_regularization_cost = L2_regularization_cost * (lambd/(2*m))
    
    total_cost = cross_entropy_cost + L2_regularization_cost

    return total_cost


def model(
    X: np.ndarray,
    Y: np.ndarray,
    layers_dims: list,
    num_iters: int = 30000,
    epsilon: float = 1e-8,
    learning_rate: float = 0.3,
    lambd: float = 0.0,
    keep_prob: float = 1.0,
    print_cost: bool = True,
    print_interval: int = 1000
) -> tuple:
    """
    Implements a deep neural network model with configurable L2 regularization and dropout.
    
    Arguments:
    X -- input data, shape: (features, number_of_examples)
    Y -- true labels, shape: (1, number_of_examples)
    layers_dims -- list of integers representing the dimensions of each layer
    num_iters -- number of iterations for optimization
    epsilon -- small constant for numerical stability
    learning_rate -- learning rate of the gradient descent update rule
    lambd -- L2 regularization parameter (0 = no regularization)
    keep_prob -- probability of keeping a neuron active during dropout (1 = no dropout)
    print_cost -- boolean, True to print cost during training
    print_interval -- interval of iterations between printing costs
    
    Returns:
    parameters -- parameters learned by the model
    costs -- list of costs recorded during training
    """
    costs = []
    
    L = len(layers_dims)
    L = L - 1  # don't count X which isn't a layer with parameters
    
    m = X.shape[1]
    
    parameters = initialize_parameters_he(layers_dims)
    
    for i in range(num_iters):
        
        # Forward propagation
        if keep_prob == 1:
            AL, caches = forward_propagation(X, parameters)
        elif keep_prob < 1:
            AL, caches = forward_propagation_with_dropout(X, parameters, keep_prob)
        
        # Compute cost
        if lambd:
            J = L2_cost(AL, Y, parameters, lambd)
        else:
            J = crossentropy_cost(AL, Y)

        # Backward propagation
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(AL, Y, caches)
        elif lambd != 0:
            grads = L2_backpropagation(AL, Y, caches, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(AL, Y, caches, keep_prob)
        
        # Update parameters using gradient descent        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # Print the loss every 10000 iterations
        if i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, J))
        if i % 1000 == 0:
            costs.append(J)

    if print_cost:
        # plot the cost
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (x1,000)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
            
    return parameters, grads, costs

def update_parameters(parameters: dict, 
                      grads: dict, 
                      learning_rate: float
                     ) -> dict:
    '''
    Updates parameters using gradients
    
    Args:
    parameters (dict): Dictionary containing network parameters
    grads (dict): Dictionary containing gradients 
    learning_rate (float): Learning rate
    
    Returns:
    dict: Updated parameters
    '''
    L = len(parameters)//2
    
    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return parameters


def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """
    L = len(parameters) // 2

    for l in range(1, L+1):
        v[f'dW{l}'] = beta * v[f'dW{l}'] + (1 - beta) * grads[f'dW{l}']
        v[f'db{l}'] = beta * v[f'db{l}'] + (1 - beta) * grads[f'db{l}']

        
        parameters[f'W{l}'] -= learning_rate * v[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * v[f'db{l}']

    return parameters, v

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    L = len(parameters) // 2

    for i in range(1, L+1):
        parameters[f'W{i}'] -= learning_rate * grads[f'dW{i}']
        parameters[f'b{i}'] -= learning_rate * grads[f'db{i}']
    
    return parameters

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.01, 
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    t -- Adam variable, counts the number of taken steps
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L+1):
        # Moving average of the gradients.
        v['dW'+str(l)] = beta1 * v['dW'+str(l)] + (1-beta1) * grads['dW'+str(l)]
        v['db'+str(l)] = beta1 * v['db'+str(l)] + (1-beta1) * grads['db'+str(l)]

        # Compute bias-corrected first moment estimate. 
        v_corrected['dW'+str(l)] = v['dW'+str(l)] / (1 - (beta1 ** t)) 
        v_corrected['db'+str(l)] = v['db'+str(l)] / (1 - (beta1 ** t))

        # Moving average of the squared gradients.        
        s['dW'+str(l)] = beta2 * s['dW'+str(l)] + (1 - beta2) * (grads['dW'+str(l)] ** 2)
        s['db'+str(l)] = beta2 * s['db'+str(l)] + (1 - beta2) * (grads['db'+str(l)] ** 2)
        
        # Compute bias-corrected second raw moment estimate
        s_corrected['dW'+str(l)] = s['dW'+str(l)] / (1 - (beta2 ** t))
        s_corrected['db'+str(l)] = s['db'+str(l)] / (1 - (beta2 ** t))

        # Update parameters
        parameters['W'+str(l)] -= learning_rate * v_corrected['dW'+str(l)]/(np.sqrt(s_corrected['dW'+str(l)]) + epsilon)
        parameters['b'+str(l)] -= learning_rate * v_corrected['db'+str(l)]/(np.sqrt(s_corrected['db'+str(l)]) + epsilon)
    
    return parameters, v, s, v_corrected, s_corrected

def model_with_optimization(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True):
    """
    3-layer neural network model which can be run in different optimizer modes.
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    optimizer -- the optimizer to be passed, gradient descent, momentum or adam
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]                   # number of training examples
    
    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += cost(AL, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(AL, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / m
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    return parameters, costs

def model_with_lrate_decay(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True, decay=None, decay_rate=1):
    """
    Neural network model which can be run in different optimizer modes and includes learning rate decay
    
    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs
    decay -- function that updates learning rate 
    decay_rate -- rate used to calculate extent of decaying in decay parameter function
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]                   # number of training examples
    lr_rates = []
    learning_rate0 = learning_rate   # the original learning rate
    
    # Initialize parameters
    parameters = initialize_parameters_he(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    # Optimization loop
    for i in range(num_epochs):
        
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0 # sum of costs over all mini batches in this epoch
        
        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            AL, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += cost(AL, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(AL, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1 # Adam counter
                parameters, v, s, _, _ = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2,  epsilon)
        cost_avg = cost_total / m

        if decay:
            learning_rate = decay(learning_rate0, i, decay_rate)
        
        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print ("Cost after epoch %i: %f" %(i, cost_avg))
            if decay:
                print("learning rate after epoch %i: %f"%(i, learning_rate))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    return parameters, costs

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    np.random.seed(1)

    caches= {}
    A_prev = X
    L = len(parameters)//2  
    
    # relu params W1 to WL-1 second last layer
    for l in range(1, L):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = forward_linear(A_prev, W, b)
        A = forward_activation(Z, activation='relu')

        # Dropout
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype('int')
        A = A * D
        A /= keep_prob

        # Store caches
        caches[f'layer{l}'] = (A_prev, Z, W, b, D)
        A_prev = A
    
    # for params WL-1 last layer No Dropout
    WL = parameters[f'W{L}']
    bL = parameters[f'b{L}']
    ZL = forward_linear(A_prev, WL, bL)
    AL = forward_activation(ZL, activation='sigmoid')
    caches[f'layer{L}'] = (A_prev, ZL, WL, bL) 
    
    return AL, caches

def backward_propagation_with_dropout(AL, Y, caches, keep_prob):    
    m = Y.shape[1]
    grads = {}
    L = len(caches)
        
    # Sigmoid layer L
    current_cache = caches[f'layer{L}']
    A_prev, Z, W, b  = current_cache
    dAL = cost_grad(AL, Y)
    grads[f'dA{L}'] = dAL
    
    dA_prev, dW, db = backprop_layer(dAL, current_cache, activation='sigmoid')
    #print(f'dA{L-1}')
    grads[f'dA{L-1}'] = dA_prev
    grads[f'dW{L}'] = dW
    grads[f'db{L}'] = db

    _, _, _, _, D = caches[f'layer{L-1}'] 
    grads[f'dA{L-1}'] *= (D / keep_prob) 
    #print(f'dA{L-1}*D{L-1}')


    for l in reversed(range(1, L)):
        #print(f'layer{l}:')
        A_prev, Z, W, b, D = caches[f'layer{l}']
        current_cache = (A_prev, Z, W, b)  # expected in backprop_layer()
        dA_prev, dW, db = backprop_layer(grads[f'dA{l}'], current_cache, activation='relu')
       
        grads[f'dA{l-1}'] = dA_prev
        #print(f'dA{l-1}')
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db

        if l > 1:
            _, _, _, _, D = caches[f'layer{l-1}'] 
            grads[f'dA{l-1}'] *= (D / keep_prob) 
            #print(f'dA{l-1}*D{l-1}')


        '''# Dropout steps
        print(f'dA{l}*D{l}')
        grads[f'dA{l}'] *= (D / keep_prob)'''


    return grads

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)  # Grading purspose
    
    m = X.shape[1]
    mini_batches = []

    # Shuffle
    indices = list(np.random.permutation(m))
    X = X[:, indices]
    Y = Y[:, indices].reshape((1,m))

    inc = mini_batch_size

    # Partition
    num_complete_mini_batches = math.floor(m/inc)

    for k in range(num_complete_mini_batches):
        mini_batch_X = X[:, k * inc : (k + 1) * inc]
        mini_batch_Y = Y[:, k * inc : (k + 1) * inc]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # end case where last batch has < mini_batch_size examples
    mini_batch_X = X[:, num_complete_mini_batches * mini_batch_size : m]
    mini_batch_Y = Y[:, num_complete_mini_batches * mini_batch_size : m]
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)

    return mini_batches

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    
    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """

    L = len(parameters) // 2
    v = {}

    for l in range(1, L+1):
        v[f'dW{l}'] = np.zeros(parameters[f'W{l}'].shape)
        v[f'db{l}'] = np.zeros(parameters[f'b{l}'].shape)
    
    return v

def initialize_adam(parameters):
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient. Initialized with zeros.
                   
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient. Initialized with zeros.
                   

    """
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L+1):
        v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)

        s['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        s['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)

    return v, s

def update_lr(learning_rate0, epoch_num, decay_rate):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer
    decay_rate -- Decay rate. Scalar

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    return learning_rate0 * (1 / (1 + (decay_rate * epoch_num)))

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, timeInterval=1000):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """

    return learning_rate0 / (1 + (decay_rate * (np.floor(epoch_num/timeInterval))))
