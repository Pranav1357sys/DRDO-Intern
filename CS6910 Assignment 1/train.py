import sys
print(sys.executable)
import os
os.environ["WANDB_MODE"] = "disabled"
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tensorflow.keras.datasets import fashion_mnist
wandb.init(project="cs6910_assignment1")
config = wandb.config
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Test images shape:", x_test.shape)
print("Test labels shape:", y_test.shape)
class_names = [
"T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot"
]
sample_images = []
sample_labels = []

for i in range(10):
    idx = np.where(y_train == i)[0][0]
    sample_images.append(x_train[idx])
    sample_labels.append(class_names[i])

images = []

for i in range(10):
    images.append(wandb.Image(sample_images[i], caption=sample_labels[i]))

wandb.log({"Fashion-MNIST Samples": images})

#initializing parameters(weights and biases)
def initialize_parameters(input_size, hidden_size, output_size, num_layers, init_type="random"):

    parameters = {}
    layer_sizes = [input_size] + [hidden_size]*num_layers + [output_size]

    for l in range(1, len(layer_sizes)):

        if init_type == "random":
            parameters["W"+str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * 0.01

        elif init_type == "Xavier":
            parameters["W"+str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(1/layer_sizes[l-1])

        parameters["b"+str(l)] = np.zeros((layer_sizes[l],1))

    return parameters

params = initialize_parameters(784,128,10,2,"Xavier")

for key in params:
    print(key, params[key].shape)

#activation functions
def activation(Z, activation_type):

    if activation_type == "identity":
        return Z

    elif activation_type == "sigmoid":
        return 1/(1+np.exp(-Z))

    elif activation_type == "tanh":
        return np.tanh(Z)

    elif activation_type == "ReLU":
        return np.maximum(0,Z)
#adding softmax 
def softmax(Z):

    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)
#derivative of activation functions
def activation_derivative(Z, activation_type):

    if activation_type == "identity":
        return np.ones_like(Z)

    elif activation_type == "sigmoid":
        s = 1/(1+np.exp(-Z))
        return s*(1-s)

    elif activation_type == "tanh":
        return 1 - np.tanh(Z)**2

    elif activation_type == "ReLU":
        return (Z > 0).astype(float)
#testing activation functions
Z = np.array([-2,-1,0,1,2])

print("Sigmoid:", activation(Z,"sigmoid"))
print("Tanh:", activation(Z,"tanh"))
print("ReLU:", activation(Z,"ReLU"))
#one hot encoding
def one_hot(Y, num_classes=10):

    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1

    return one_hot_Y
#testing one hot encoding
print(one_hot(np.array([3,1,2])))
#flattening images
X_train_flat = x_train.reshape(x_train.shape[0], -1).T / 255
X_test_flat = x_test.reshape(x_test.shape[0], -1).T / 255

Y_train_oh = one_hot(y_train)
Y_test_oh = one_hot(y_test)

print(X_train_flat.shape)
print(Y_train_oh.shape)
#mini batch creation
def create_mini_batches(X, Y, batch_size):

    m = X.shape[1]
    permutation = np.random.permutation(m)

    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    mini_batches = []

    for i in range(0, m, batch_size):

        X_batch = X_shuffled[:, i:i+batch_size]
        Y_batch = Y_shuffled[:, i:i+batch_size]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
#accuracy function
def compute_accuracy(Y_hat, Y):

    predictions = np.argmax(Y_hat, axis=0)
    labels = np.argmax(Y, axis=0)

    return np.mean(predictions == labels)
#cross entorpy loss
def compute_loss(Y_hat, Y):

    m = Y.shape[1]

    loss = -np.sum(Y * np.log(Y_hat + 1e-9)) / m

    return loss
#testing loss function
Y_hat = np.array([[0.1],[0.7],[0.2]])
Y = np.array([[0],[1],[0]])

print(compute_loss(Y_hat,Y))
#forward propogation
def forward_propagation(X, parameters, activation_type):

    caches = {}
    A = X
    L = len(parameters)//2

    for l in range(1, L):

        W = parameters["W"+str(l)]
        b = parameters["b"+str(l)]

        Z = np.dot(W,A) + b
        A = activation(Z, activation_type)

        caches["Z"+str(l)] = Z
        caches["A"+str(l)] = A

    # Output layer
    W = parameters["W"+str(L)]
    b = parameters["b"+str(L)]

    Z = np.dot(W,A) + b
    A = softmax(Z)

    caches["Z"+str(L)] = Z
    caches["A"+str(L)] = A

    return A, caches

#test
X_sample = np.random.randn(784,1)

params = initialize_parameters(784,128,10,2,"Xavier")

output, cache = forward_propagation(X_sample, params, "ReLU")

print(output.shape)
print(np.sum(output))
#backpropogation
def backpropagation(X, Y, parameters, caches, activation_type):

    grads = {}
    m = X.shape[1]
    L = len(parameters)//2

    A_final = caches["A"+str(L)]

    # Output layer gradient
    dZ = A_final - Y

    for l in reversed(range(1, L+1)):

        if l == 1:
            A_prev = X
        else:
            A_prev = caches["A"+str(l-1)]

        W = parameters["W"+str(l)]

        grads["dW"+str(l)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db"+str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        if l > 1:

            Z_prev = caches["Z"+str(l-1)]
            dA_prev = np.dot(W.T, dZ)

            dZ = dA_prev * activation_derivative(Z_prev, activation_type)

    return grads
#testing backpropagation
X_sample = np.random.randn(784,5)

params = initialize_parameters(784,128,10,2,"Xavier")

Y_sample = np.zeros((10,5))
Y_sample[3] = 1

output, cache = forward_propagation(X_sample, params, "ReLU")

grads = backpropagation(X_sample, Y_sample, params, cache, "ReLU")

for g in grads:
    print(g, grads[g].shape)
#stochastic gradient descent
def update_parameters_sgd(parameters, grads, learning_rate):

    L = len(parameters)//2

    for l in range(1, L+1):

        parameters["W"+str(l)] -= learning_rate * grads["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate * grads["db"+str(l)]

    return parameters
#testing sgd
params = initialize_parameters(784,128,10,2,"Xavier")

X_sample = np.random.randn(784,5)

Y_sample = np.zeros((10,5))
Y_sample[3] = 1

output, cache = forward_propagation(X_sample, params, "ReLU")

grads = backpropagation(X_sample, Y_sample, params, cache, "ReLU")

params = update_parameters_sgd(params, grads, 0.01)

print("Update completed")
#momnetum based gradient descent
def initialize_velocity(parameters):

    v = {}
    L = len(parameters)//2

    for l in range(1, L+1):

        v["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
        v["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    return v

def update_parameters_momentum(parameters, grads, v, learning_rate, beta=0.9):

    L = len(parameters)//2

    for l in range(1, L+1):

        v["dW"+str(l)] = beta*v["dW"+str(l)] + (1-beta)*grads["dW"+str(l)]
        v["db"+str(l)] = beta*v["db"+str(l)] + (1-beta)*grads["db"+str(l)]

        parameters["W"+str(l)] -= learning_rate * v["dW"+str(l)]
        parameters["b"+str(l)] -= learning_rate * v["db"+str(l)]

    return parameters, v
#initialization of RMSprop
def initialize_rmsprop(parameters):

    s = {}
    L = len(parameters)//2

    for l in range(1, L+1):

        s["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
        s["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    return s
#rmsprop update
def update_parameters_rmsprop(parameters, grads, s, learning_rate, beta=0.9, epsilon=1e-8):

    L = len(parameters)//2

    for l in range(1, L+1):

        s["dW"+str(l)] = beta*s["dW"+str(l)] + (1-beta)*(grads["dW"+str(l)]**2)
        s["db"+str(l)] = beta*s["db"+str(l)] + (1-beta)*(grads["db"+str(l)]**2)

        parameters["W"+str(l)] -= learning_rate * grads["dW"+str(l)] / (np.sqrt(s["dW"+str(l)]) + epsilon)
        parameters["b"+str(l)] -= learning_rate * grads["db"+str(l)] / (np.sqrt(s["db"+str(l)]) + epsilon)

    return parameters, s
#adam variables initialization
def initialize_adam(parameters):

    v = {}
    s = {}

    L = len(parameters)//2

    for l in range(1, L+1):

        v["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
        v["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

        s["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
        s["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    return v, s
#adam update
def update_parameters_adam(parameters, grads, v, s, t, learning_rate,
                           beta1=0.9, beta2=0.999, epsilon=1e-8):

    L = len(parameters)//2

    for l in range(1, L+1):

        v["dW"+str(l)] = beta1*v["dW"+str(l)] + (1-beta1)*grads["dW"+str(l)]
        v["db"+str(l)] = beta1*v["db"+str(l)] + (1-beta1)*grads["db"+str(l)]

        s["dW"+str(l)] = beta2*s["dW"+str(l)] + (1-beta2)*(grads["dW"+str(l)]**2)
        s["db"+str(l)] = beta2*s["db"+str(l)] + (1-beta2)*(grads["db"+str(l)]**2)

        v_corrected_dW = v["dW"+str(l)] / (1-beta1**t)
        v_corrected_db = v["db"+str(l)] / (1-beta1**t)

        s_corrected_dW = s["dW"+str(l)] / (1-beta2**t)
        s_corrected_db = s["db"+str(l)] / (1-beta2**t)

        parameters["W"+str(l)] -= learning_rate * v_corrected_dW / (np.sqrt(s_corrected_dW)+epsilon)
        parameters["b"+str(l)] -= learning_rate * v_corrected_db / (np.sqrt(s_corrected_db)+epsilon)

    return parameters, v, s
#test adam
params = initialize_parameters(784,128,10,2,"Xavier")

v, s = initialize_adam(params)

X_sample = np.random.randn(784,5)

Y_sample = np.zeros((10,5))
Y_sample[3] = 1

output, cache = forward_propagation(X_sample, params, "ReLU")

grads = backpropagation(X_sample, Y_sample, params, cache, "ReLU")

params, v, s = update_parameters_adam(params, grads, v, s, t=1, learning_rate=0.001)

print("Adam working")
#implement NAG
def update_parameters_nag(parameters, grads, v, learning_rate, beta=0.9):

    L = len(parameters)//2

    for l in range(1, L+1):

        v_prev_dW = v["dW"+str(l)]
        v_prev_db = v["db"+str(l)]

        v["dW"+str(l)] = beta*v["dW"+str(l)] + (1-beta)*grads["dW"+str(l)]
        v["db"+str(l)] = beta*v["db"+str(l)] + (1-beta)*grads["db"+str(l)]

        parameters["W"+str(l)] -= learning_rate*(beta*v["dW"+str(l)] + (1-beta)*grads["dW"+str(l)])
        parameters["b"+str(l)] -= learning_rate*(beta*v["db"+str(l)] + (1-beta)*grads["db"+str(l)])

    return parameters, v
#nadam updaate
def update_parameters_nadam(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):

    L = len(parameters)//2

    for l in range(1, L+1):

        v["dW"+str(l)] = beta1*v["dW"+str(l)] + (1-beta1)*grads["dW"+str(l)]
        v["db"+str(l)] = beta1*v["db"+str(l)] + (1-beta1)*grads["db"+str(l)]

        s["dW"+str(l)] = beta2*s["dW"+str(l)] + (1-beta2)*(grads["dW"+str(l)]**2)
        s["db"+str(l)] = beta2*s["db"+str(l)] + (1-beta2)*(grads["db"+str(l)]**2)

        v_corrected_dW = v["dW"+str(l)] / (1-beta1**t)
        v_corrected_db = v["db"+str(l)] / (1-beta1**t)

        s_corrected_dW = s["dW"+str(l)] / (1-beta2**t)
        s_corrected_db = s["db"+str(l)] / (1-beta2**t)

        nadam_dW = beta1*v_corrected_dW + (1-beta1)*grads["dW"+str(l)]/(1-beta1**t)
        nadam_db = beta1*v_corrected_db + (1-beta1)*grads["db"+str(l)]/(1-beta1**t)

        parameters["W"+str(l)] -= learning_rate * nadam_dW/(np.sqrt(s_corrected_dW)+epsilon)
        parameters["b"+str(l)] -= learning_rate * nadam_db/(np.sqrt(s_corrected_db)+epsilon)

    return parameters, v, s
#training loop
#training loop
def train_network(X, Y, parameters, epochs, batch_size, learning_rate,
                  activation_type, optimizer="sgd"):

    L = len(parameters)//2

    v, s = None, None
    t = 1

    if optimizer in ["momentum","nag"]:
        v = initialize_velocity(parameters)

    if optimizer in ["rmsprop"]:
        s = initialize_rmsprop(parameters)

    if optimizer in ["adam","nadam"]:
        v, s = initialize_adam(parameters)

    for epoch in range(epochs):

        mini_batches = create_mini_batches(X,Y,batch_size)

        epoch_loss = 0

        for X_batch, Y_batch in mini_batches:

            Y_hat, caches = forward_propagation(X_batch, parameters, activation_type)

            loss = compute_loss(Y_hat, Y_batch)

            grads = backpropagation(X_batch, Y_batch, parameters, caches, activation_type)

            if optimizer == "sgd":

                parameters = update_parameters_sgd(parameters, grads, learning_rate)

            elif optimizer == "momentum":

                parameters, v = update_parameters_momentum(parameters, grads, v, learning_rate)

            elif optimizer == "nag":

                parameters, v = update_parameters_nag(parameters, grads, v, learning_rate)

            elif optimizer == "rmsprop":

                parameters, s = update_parameters_rmsprop(parameters, grads, s, learning_rate)

            elif optimizer == "adam":

                parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, learning_rate)

            elif optimizer == "nadam":

                parameters, v, s = update_parameters_nadam(parameters, grads, v, s, t, learning_rate)

            epoch_loss += loss
            t += 1

        epoch_loss /= len(mini_batches)

        Y_hat_full, _ = forward_propagation(X, parameters, activation_type)
        acc = compute_accuracy(Y_hat_full, Y)

        wandb.log({"epoch":epoch, "loss":epoch_loss, "accuracy":acc})

        val_acc = evaluate_model(X_val, Y_val, parameters, activation_type)
        wandb.log({
            "epoch": epoch,
            "loss": epoch_loss,
            "train_accuracy": acc,
            "val_accuracy": val_acc
        })

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")
        print(f"Epoch {epoch+1}, Loss:{epoch_loss:.4f}, Train Acc:{acc:.4f}, Val Acc:{val_acc:.4f}")

    return parameters

#validation split
split = 54000

X_val = X_train_flat[:, split:]
Y_val = Y_train_oh[:, split:]

X_train_new = X_train_flat[:, :split]
Y_train_new = Y_train_oh[:, :split]

print(X_train_new.shape)
print(X_val.shape)
#validation accuracy function
def evaluate_model(X, Y, parameters, activation):

    Y_hat, _ = forward_propagation(X, parameters, activation)

    acc = compute_accuracy(Y_hat, Y)

    return acc

#sweep entry function
def sweep_train():

    wandb.init()
    config = wandb.config

    params = initialize_parameters(784, config.hidden_size, 10, 2, "Xavier")

    trained_params = train_network(
        X_train_flat,
        Y_train_oh,
        params,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        activation_type=config.activation,
        optimizer=config.optimizer
    )

    val_acc = evaluate_model(X_val, Y_val, trained_params, config.activation)

    wandb.log({"val_accuracy": val_acc})

    if __name__ == "__main__":
        sweep_train()