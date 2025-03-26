import numpy as np
import math
import nn_from_scratch_functions as nn
import matplotlib.pyplot as plt

def flatten_parameters(parameters):
    """
    Flatten parameters dictionary into a single vector.

    """
    theta = np.array([]).reshape(-1,1)

    for key in parameters.keys():
        new_vector = parameters[key].reshape((-1, 1))
        theta = np.concatenate((theta, new_vector), axis=0)

    return theta
    
def unflatten_parameters(theta, parameter_shapes, prefix=''):
    """
    Unflatten vector back into the parameters dictionary.
    
    Args:
    theta (ndarray)(1,): flattened parameters
    parameter_shapes (dict): Shapes of parameters, {W1 : W1.shape, ...} for all parameters
    prefix (string): Prefix to apply to keys when unrolled
    
    Returns:
    parameters (dict): Unflattened dictionary {dW1, db1,...}
    """
    parameters = {}
    start = 0

    for key in parameter_shapes.keys():
        shape = parameter_shapes[key]
        size = np.prod(shape)
        parameters[f'{prefix}{key}'] = theta[start:start + size].reshape(shape)
        start += size

    return parameters

def order_parameters(parameters):
    '''
    Orders parameters in ascending order as  dW1, db1, dW2, db2, ...
    '''
    tmp = {}
    for i in range(1, (len(parameters)//2) + 1):
        tmp[f'dW{i}'] = parameters[f'dW{i}']
        tmp[f'db{i}'] = parameters[f'db{i}']
    return tmp    
    
def compute_numerical_gradient(X, Y, 
                               parameters, 
                               forward_propagation, 
                               forwardprop_args,
                               cost_function: str, 
                               cost_args, 
                               epsilon=1e-7):
    """
    Compute the numerical gradient for gradient checking.
    parameters (dict): Contains Ws and bs
    """
    theta = flatten_parameters(parameters)
    
    parameter_shapes = {key: parameters[key].shape for key in parameters.keys()}
    
    # # parameters including those in Ws and bs
    num_total_parameters = theta.shape[0]
   
    # array of numerical gradient of each parameter
    numerical_gradients = np.zeros((num_total_parameters, 1))
    
    J_plus = np.zeros((num_total_parameters, 1))
    J_minus = np.zeros((num_total_parameters, 1))

    for i in range(num_total_parameters):
        theta_plus = np.copy(theta)
        theta_minus = np.copy(theta)

        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon
        
        params_plus = unflatten_parameters(theta_plus, parameter_shapes)
        params_minus = unflatten_parameters(theta_minus, parameter_shapes)

        AL_plus, _ = forward_propagation(X, params_plus, **forwardprop_args)
        AL_minus, _ = forward_propagation(X, params_minus, **forwardprop_args)


        if cost_function == 'l2_cost':
            J_plus[i] = nn.L2_cost(AL_plus, Y, params_plus, **cost_args)
            J_minus[i] = nn.L2_cost(AL_minus, Y, params_minus, **cost_args)
        elif cost_function == 'crossentropy_cost':
            J_plus[i] = nn.crossentropy_cost(AL_plus, Y, **cost_args)
            J_minus[i] = nn.crossentropy_cost(AL_minus, Y, **cost_args)
        
        numerical_gradients[i] = (J_plus[i] - J_minus[i]) / (2*epsilon)

    return unflatten_parameters(numerical_gradients, parameter_shapes, prefix='d')


def gradient_checking(X, Y, 
                      parameters, 
                      forward_propagation,
                      backward_propagation, 
                      forwardprop_args={},
                      backprop_args = {},
                      cost_function: str= 'crossentropy_cost', 
                      cost_args={}):

    gradapprox = compute_numerical_gradient(X, Y, 
                                            parameters, 
                                            forward_propagation, 
                                            forwardprop_args,
                                            cost_function, 
                                            cost_args)
        
    AL, caches = forward_propagation(X, parameters, **forwardprop_args)
    grads = backward_propagation(AL, Y, caches, **backprop_args)

    # Filter out non-parameter gradients dA
    keys_to_remove = [key for key in grads if key.startswith("dA")]
    for key in keys_to_remove:
        del grads[key]
        

    # order of keys are different in grads and gradapprox. imrove this later.
    # grads store derivatives in reverse order due to backprop db3, dW3, ...
    # Order keys of grads before flattening it
    grads = order_parameters(grads)
  
    gradapprox = flatten_parameters(gradapprox)
    grads = flatten_parameters(grads)

    # Difference
    numerator = np.linalg.norm(grads - gradapprox)
    denominator = np.linalg.norm(grads) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if True:       
        if difference > 1e-5:
            print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

def predict(X, y, parameters, forward_propagation):
    """
    This function is used to predict the results of a  n-layer neural network.
    Args:
    X (ndarray): data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    m = X.shape[1]
    L = len(parameters)//2
    AL, caches = forward_propagation(X, parameters)
    p = np.int64(AL > 0.5)
    print(f'Accuracy: {np.mean(p == y)}')
    
    return p



def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    AL, cache = nn.forward_propagation(X, parameters)
    predictions = (AL > 0.5)
    return predictions

