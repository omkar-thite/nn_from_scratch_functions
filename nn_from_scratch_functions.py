def initialize_parameters(layers_dims):
    L = len(layers_dims)    
    parameters = {}
    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) / np.sqrt(layers_dims[l-1])
        parameters[f'b{l}'] = np.zeros((layers_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layers_dims[l], layers_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layers_dims[l], 1))
    
    '''print(f'Initialized parameters:')
    for key in parameters:
        print(f'{key} shape: {parameters[key].shape}')'''
        
    return parameters

def forward_linear(A_prev, W, b):
    Z = np.dot(W, A_prev) + b

    #print(f'Calculated Z, shape: {Z.shape}')
    
    return Z

def forward_activation(Z, activation):
    if activation=='relu':
        A = relu(Z)
        #print(f'Calculated A=relu(Z), shape: {A.shape}')
        return A
    elif activation=='sigmoid':
        A = sigmoid(Z)
        #print(f'Calculated A = sigmoid(Z): {A.shape}')
        return A
    else:
        return Z

def cost(AL, Y, epsilon=1e-8):
    J = - np.sum((Y * np.log(AL+ epsilon)) + (1 - Y) * np.log(1-AL + epsilon)) / Y.shape[1]
   # print(f'Calculated J = {J}, J type: {type(J)}')
    return np.squeeze(J)

def cost_grad(AL, Y, epsilon=1e-8):
    dAL = -(np.divide(Y, AL+epsilon) - np.divide(1 - Y, 1 - AL + epsilon))
   # print(f'Calculated dAL, shape: {dAL.shape}')
    return dAL

def d_Z(dA, Z, activation):
    if activation == 'relu':
        dZ = dA * relu_prime(Z)
        #print(f'Calculated dZ, relu layer, shape:{dZ.shape}')
        return dZ
    elif activation == 'sigmoid':
        dZ = dA * sigmoid_prime(Z)
       # print(f'Calculated dZ, sigmoid layer, shape:{dZ.shape}')
        return dZ

def relu_prime(Z):
    g_prime = np.where(Z >= 0, 1, 0)
    #print(f'Calculated relu prime, shape: {g_prime.shape}')
    return g_prime

def relu(Z):
    A = np.maximum(0, Z)
    return A

def sigmoid_prime(Z):
    a = sigmoid(Z)
    g_prime = a * (1 - a)
   # print(f'Calculated sigmoid prime, shape: {g_prime.shape}')    
    return g_prime

def sigmoid(Z):
    A = 1/(1 + np.exp(-Z))
    return A
    
def forward_propagation(X, parameters): 
    caches= {}
    A_prev = X
    L = len(parameters)//2
    
    # relu params W1 to WL-1 second last layer
    for l in range(1, L):
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = forward_linear(A_prev, W, b)
        A = forward_activation(Z, activation='relu')
        caches[f'layer{l}'] = (A_prev, Z, W, b)
        A_prev = A
    
    # for params WL-1 last layer
    WL = parameters[f'W{L}']
    bL = parameters[f'b{L}']
    ZL = forward_linear(A_prev, WL, bL)
    AL = forward_activation(ZL, activation='sigmoid')
    caches[f'layer{L}'] = (A_prev, ZL, WL, bL) 
    
    return AL, caches


def backprop_layer(dA, cache, activation):
    m = dA.shape[1]
    A_prev, Z, W, b = cache
    
    dZ = d_Z(dA, Z, activation)
    dA_prev =  np.dot(W.T, dZ) 
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ,axis=1, keepdims=True) / m
    
    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    #print('backprop started...')
    
    m = Y.shape[1]
    grads = {}
    L = len(caches)
    
    #print(f'm={m}, L={L}')
    
    # Sigmoid layer L
    current_cache = caches[f'layer{L}']
    A_prev, Z, W, b = current_cache
    dAL = cost_grad(AL, Y)
    grads[f'dA{L}'] = dAL
    #print(f"grads[f'dA{L}'] done with dA{L} shape: {grads[f'dA{L}'].shape}")
    
    dA_prev, dW, db = backprop_layer(dAL, current_cache, activation='sigmoid')
    grads[f'dA{L-1}'] = dA_prev
    grads[f'dW{L}'] = dW
    grads[f'db{L}'] = db
    
    for l in range(L-1, 0, -1):
              
        current_cache = caches[f'layer{l}']
        A_prev, Z, W, b = current_cache 
        dA_prev, dW, db = backprop_layer(grads[f'dA{l}'], current_cache, activation='relu')
        grads[f'dA{l-1}'] = dA_prev
        grads[f'dW{l}'] = dW
        grads[f'db{l}'] = db
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2

    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']
    return parameters

def L2_backpropagation(AL, Y, caches, lambd):
    L = len(caches)
    m = Y.shape[1]
    grads = backward_propagation(AL, Y, caches)
    
    for l in range(1, L+1):
        (A_prev, Z, W, b) = caches[f'layer{l}']
        grads[f'dW{l}'] += (lambd/m) * W
    return grads

def L2_cost(AL, Y, parameters, lambd):
    m = Y.shape[1]
    L = len(parameters)//2
    cross_entropy_cost = cost(AL, Y)

    L2_regularization_cost = 0
    for l in range(1, L+1):
        L2_regularization_cost += np.sum(np.square(parameters[f'W{l}']))
    
    L2_regularization_cost = L2_regularization_cost * (lambd/(2*m))
    
    total_cost = cross_entropy_cost + L2_regularization_cost

    return total_cost


def model(X, Y, layers_dims, num_iters=30000, epsilon=1e-8, learning_rate=0.3, lambd=0.0, keep_prob=1):
    costs = []
    L = len(layers_dims)
    L = L - 1 # remove X which isn't layer
    m = X.shape[1]
    
    parameters = initialize_parameters(layers_dims)
    
    for i in range(num_iters):

        if keep_prob == 1:
            AL, caches = forward_propagation(X, parameters)
        elif keep_prob < 1:
            AL, caches = forward_propagation_with_dropout(X, parameters, keep_prob)

       # AL = np.clip(AL, epsilon, 1 - epsilon)
        
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(AL, Y, caches)
        elif lambd != 0:
            grads = L2_backpropagation(AL, Y, caches, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(AL, Y, caches, keep_prob)

            
        parameters = update_parameters(parameters, grads, learning_rate)

         # Print the loss every 10000 iterations
        if lambd:
            J = L2_cost(AL, Y, parameters, lambd)
        else:
            J = cost(AL, Y)
            
        if i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, J))
        if i % 1000 == 0:
            costs.append(J)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    

    return parameters, grads

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Args:
    parameters (dict): parameters                
    grads (dict): gradients             
    v (dict): current velocity:
                 
    beta (scalar): momentum hyperparameter
    learning_rate (scalar): learning rate
    
    Returns:
    parameters (dict): updated parameters 
    v (dict): updated velocities
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
    
    Args:
    parameters (dict): parameters                
    grads (dict): gradients                      
    learning_rate (scalar): learning rate
    
    Returns:
    parameters (dict): updated parameters 
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
    parameters (dict): parameters                
    grads (dict): gradients           
    
    v (dict): moving average of the first gradient (Adam variable)
    s (dict): moving average of the squared gradient (Adam variable)
    t (scalar): number of taken steps (Adam variable)
    
    learning_rate (scalar): learning rate
    
    beta1 (scalar): Exponential decay hyperparameter for the first moment estimates 
    beta2 (scalar): Exponential decay hyperparameter for the second moment estimates 
    epsilon (scalar): Hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters (dict): updated parameters 
    v (dict): moving average of the first gradient (Adam variable)
    s (dict): moving average of the squared gradient (Adam variable)
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


def model_with_lrate_decay(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 5000, print_cost = True, decay=None, decay_rate=1):
    """
    Neural network model which can be run in different optimizer modes and includes learning rate decay
    
    Args:
    X (ndarray) (2, number of examples): input data
    Y (ndarray) (1, number of examples):  True label vector (1 for blue dot / 0 for red dot)
    layers_dims (list): Specifies size of each layer
    learning_rate (scalar): learning rate
    mini_batch_size (scalar): size of a mini batch
    beta (scalar): Momentum hyperparameter
    beta1 (scalar): Exponential decay hyperparameter for the first moment estimates 
    beta2 (scalar): Exponential decay hyperparameter for the second moment estimates 
    epsilon (scalar): Hyperparameter preventing division by zero in Adam updates
    num_epochs (int): number of epochs
    print_cost (bool): True to print the cost every 1000 epochs
    decay (callable): function that updates learning rate 
    decay_rate (scalar): rate used to calculate extent of decaying in decay parameter function
    
    Returns:
    parameters (dict): updated parameters 
    """

    L = len(layers_dims)             # number of layers in the neural networks
    costs = []                       # to keep track of the cost
    t = 0                            # initializing the counter required for Adam update
    seed = 10                        # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]                   # number of training examples
    lr_rates = []
    learning_rate0 = learning_rate   # the original learning rate
    
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
                
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
