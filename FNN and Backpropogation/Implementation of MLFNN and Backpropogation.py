import numpy as np

class MLFFNN:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.L = len(layer_sizes) - 1
        self.lr = learning_rate
        self.W = {}
        self.b = {}
        for l in range(1, len(layer_sizes)):
            self.W[l] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * np.sqrt(2 / layer_sizes[l-1])
            self.b[l] = np.zeros((layer_sizes[l], 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward(self, X):
        self.A = {0: X}
        self.Z = {}
        for l in range(1, self.L):
            self.Z[l] = np.dot(self.W[l], self.A[l-1]) + self.b[l]
            self.A[l] = self.relu(self.Z[l])
        self.Z[self.L] = np.dot(self.W[self.L], self.A[self.L-1]) + self.b[self.L]
        self.A[self.L] = self.softmax(self.Z[self.L])
        return self.A[self.L]

    def compute_loss(self, Y, Y_hat):
        m = Y.shape[1]
        return -(1/m) * np.sum(Y * np.log(Y_hat + 1e-8))

    def backward(self, Y):
        m = Y.shape[1]
        self.dW = {}
        self.db = {}
        self.dZ = {}
        self.dZ[self.L] = self.A[self.L] - Y
        for l in reversed(range(1, self.L + 1)):
            self.dW[l] = (1/m) * np.dot(self.dZ[l], self.A[l-1].T)
            self.db[l] = (1/m) * np.sum(self.dZ[l], axis=1, keepdims=True)
            if l > 1:
                self.dZ[l-1] = np.dot(self.W[l].T, self.dZ[l]) * self.relu_derivative(self.Z[l-1])

    def update_parameters(self):
        for l in range(1, self.L + 1):
            self.W[l] -= self.lr * self.dW[l]
            self.b[l] -= self.lr * self.db[l]

    def train(self, X, Y, epochs=5000):
        for i in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            self.backward(Y)
            self.update_parameters()
            if i % 500 == 0:
                print("Epoch:", i, "Loss:", loss)

    def predict(self, X):
        Y_hat = self.forward(X)
        return np.argmax(Y_hat, axis=0)


X = np.array([[0,0,1,1],
              [0,1,0,1]])

Y_labels = np.array([0,1,1,0])
Y = np.zeros((2,4))
Y[Y_labels, np.arange(4)] = 1

nn = MLFFNN([2,8,2], learning_rate=0.1)
nn.train(X, Y, epochs=5000)

predictions = nn.predict(X)

print("Predictions:", predictions)
print("Actual:", Y_labels)