---
title: "Building a neural network from scratch"
categories:
  - Deep-learning
tags:
  - Deep learning
  - Neural Networks
  - python
---

### Code your neural network today in python from scratch
We will build a neural network from scratch and its different components like forward propogarion, gradient descent, back propogation. Fire up your favourite editor and start your journey to the world of neural nets.

{% highlight python linenos %}
# Coding the forward propagation algorithm
input_data = np.array([3,5])
weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 'output': np.array([2, 7])}

# Calculate node 0 value: node_0_value
node_0_value = (input_data * weights['node_0']).sum()

# Calculate node 1 value: node_1_value
node_1_value = node_1_value = (input_data * weights['node_1']).sum()

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_value, node_1_value])

# Calculate output: output
output = (hidden_layer_outputs * weights['output']).sum()

# Print output
print(output)
{% endhighlight %}

This basic neural network can only linear relation in data. To capture non-linearity in data, we need a non-linear activation function. Some popular activation functions are sigmoid, tanh and relu. RELU or Rectified linear units work very well in neural networks and we will be using them.

{% highlight python linenos %}
# The Rectified Linear Activation Function
#  an "activation function" is a function applied at each node. It converts the node's input into some output.
def relu(input):
    '''Define your relu activation function here'''
    # Calculate the value for the output of the relu function: output
    output = max(input, 0)
    
    # Return the value just calculated
    return(output)

# Calculate node 0 value: node_0_output
node_0_input = (input_data * weights['node_0']).sum()
node_0_output = relu(node_0_input)

# Calculate node 1 value: node_1_output
node_1_input = (input_data * weights['node_1']).sum()
node_1_output = relu(node_1_input)

# Put node values into array: hidden_layer_outputs
hidden_layer_outputs = np.array([node_0_output, node_1_output])

# Calculate model output (do not apply relu)
model_output = (hidden_layer_outputs * weights['output']).sum()

# Print model output
print(model_output)

# Without this activation function, we would have predicted a negative number! The real power of activation functions will come soon when we start tuning model weights.
{% endhighlight %}

The deeper networks learn better representation of data and remove the need to manually learn the features. Let now work with more data. 

{% highlight python linenos %}
# Applying the network to many observations/rows of data
input_data = np.array([[3, 5], [1, -1], [0, 0], [8, 4]])
weights = {'node_0': np.array([2, 4]), 'node_1': np.array([ 4, -5]), 'output': np.array([2, 7])}

# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)


# Create empty list to store prediction results
results = []
for input_data_row in input_data:
    # Append prediction to results
    results.append(predict_with_network(input_data_row, weights))

# Print results
print(results)
{% endhighlight %}

Let try to build a neural network with two hidden layers each containing two nodes. In deeper architectures, it is generally the last layer that captures the most complex interactions.

{% highlight python linenos %}
input_data = np.array([3, 5])
weights = {'node_0_0': np.array([2, 4]), 'node_0_1': np.array([ 4, -5]), 'node_1_0': np.array([-1,  1]),
 'node_1_1': np.array([2, 2]), 'output': np.array([2, 7])}

def predict_with_network(input_data):
    # Calculate node 0 in the first hidden layer
    node_0_0_input = (input_data * weights['node_0_0']).sum()
    node_0_0_output = relu(node_0_0_input)

    # Calculate node 1 in the first hidden layer
    node_0_1_input = (input_data * weights['node_0_1']).sum()
    node_0_1_output = relu(node_0_1_input)

    # Put node values into array: hidden_0_outputs
    hidden_0_outputs = np.array([node_0_0_output, node_0_1_output])
    
    # Calculate node 0 in the second hidden layer
    node_1_0_input = (hidden_0_outputs * weights['node_1_0']).sum()
    node_1_0_output = relu(node_1_0_input)

    # Calculate node 1 in the second hidden layer
    node_1_1_input = (hidden_0_outputs * weights['node_1_1']).sum()
    node_1_1_output = relu(node_1_1_input)

    # Put node values into array: hidden_1_outputs
    hidden_1_outputs = np.array([node_1_0_output, node_1_1_output])

    # Calculate model output: model_output
    model_output = (hidden_1_outputs * weights['output']).sum()
    
    # Return model_output
    return(model_output)

output = predict_with_network(input_data)
print(output)
# The network generated a prediction of 364
{% endhighlight %}

Let now see how changing the weights will affect our output.

{% highlight python linenos %}
# Calculating model errors
# Coding how weight changes affect accuracy

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }
# Define predict_with_network()
def predict_with_network(input_data_row, weights):

    # Calculate node 0 value
    node_0_input = (input_data_row * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 value
    node_1_input = (input_data_row * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_layer_outputs
    hidden_layer_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output
    input_to_final_layer = (hidden_layer_outputs * weights['output']).sum()
    model_output = relu(input_to_final_layer)
    
    # Return model output
    return(model_output)

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [-1, 1]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)
# The network now generates a perfect prediction with an error of 0.
{% endhighlight %}

Let now do it for multiple data points

{% highlight python linenos %}
# Scaling up to multiple data points
input_data = np.array([[0, 3], [1, 2], [-1, -2], [4, 0]])

weights_0 = {'node_0': np.array([2, 1]), 'node_1': np.array([1, 2]), 'output': np.array([1, 1])}
weights_1 = {'node_0': np.array([2, 1]), 'node_1': np.array([ 1. ,  1.5]), 'output': np.array([ 1. ,  1.5])}

target_actuals = [1, 3, 5, 7] 

#from sklearn.metrics import mean_squared_error

# Create model_output_0 
model_output_0 = []
# Create model_output_0
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))


# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(target_actuals, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" %mse_0)
print("Mean squared error with weights_1: %f" %mse_1)
# It looks like model_output_1 has a higher mean squared error.
{% endhighlight %}

We just saw how changing weights can change our output. A loss function generally aggregates all the errors and gives us a single value. In regression problems, we use mse, rmse, mape to compute the error. The algorithm to do this is called Gradient Descent. It will give us the set of weights that gives the lowest error. Mathematically, we move in direction opposite to the slope(derivative) of tangent, this captures the slope of our loss function, so its called gradient descent. It allows us to move slowly in a direction where the loss is the minimum. Now we will see how to compute the slope of a set of data points.

{% highlight python linenos %}
# Calculating slopes 
# 1) slope of loss function wrt value at node we feed into 
# 2) value of node that feeds into weight 
# 3) slope of activation function wrt value we feed into
# When plotting the mean-squared error loss function against predictions, the slope is 2 * x * (y-xb), or 2 * input_data * error. 
# Note that x and b may have multiple numbers (x is a vector for each data point, and b is a vector)

weights = np.array([0,2,1])
input_data = np.array([1,2,3])
target = 0

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Print the slope
print(slope)
# This slope can be used to improve the weights of the model.
{% endhighlight %}

Lets use the slope calculated to update our model weights. The learning rate tells us how fast or slow we move towards the lowest error point. This is also one of the hyperparameter to be tuned for good performance. 

{% highlight python linenos %}
# Improving model weights
weights = np.array([0, 2, 1])
input_data = np.array([1,2,3])
target = 0

# Set the learning rate: learning_rate
learning_rate = 0.01

# Calculate the predictions: preds
preds = (weights * input_data).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = 2 * input_data * error

# Update the weights: weights_updated
weights_updated = weights - slope * learning_rate

# Get updated predictions: preds_updated
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error: error_updated
error_updated = preds_updated - target

# Print the original error
print(error)

# Print the updated error
print(error_updated)
# Updating the model weights did decrease the error.
{% endhighlight %}

Let clean up our code and build functions to compute slope, error and weight updates. And lets see how multiple weight updates improve the mean squared error.

{% highlight python linenos %}
# Making multiple updates to weights
def get_slope(input_data, target, weights):
    preds = (weights * input_data).sum()
    error = preds - target
    slope = 2 * input_data * error    
    return slope

def get_mse(input_data, target, weights):
    learning_rate = 0.01
    preds = (weights * input_data).sum()
    error = preds - target
    slope = 2 * input_data * error
    weights_updated = weights - slope * learning_rate
    preds_updated = (weights_updated * input_data).sum()
    error_updated = preds_updated - target
    return error_updated

weights = np.array([0, 2, 1])
input_data = np.array([1,2,3])
target = 0

n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope: slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights: weights
    weights = weights - 0.01 * slope
    
    # Calculate mse with new weights: mse
    mse = get_mse(input_data, target, weights)
    #print("iteration {}, weights {}, mse {}".format(i, weights, mse))
    # Append the mse to mse_hist
    mse_hist.append(mse)

# Plot the mse history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()
# As you can see, the mean squared error decreases as the number of iterations go up.
{% endhighlight %}
![mse1]({{ site.url }}/reflections/assets/images/201707271.png)

You just used your error to update the model. This is called back propogation. Now since you have a solid understanding of neural network, you can code this using keras with couple of lines. Recently Geoff hinton criticise his work by saying "throw it away and start all over again" in search of AI that actually learn like humans with no labels with less examples. Synthetic gradients, used by [deep mind](https://deepmind.com/){:target="_blank"}  to compute the loss function is also worth taking a look. 