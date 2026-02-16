import numpy as np

# -----------------------------
# Sample Dataset
# -----------------------------

X_train = np.array([
    [1,2],
    [2,3],
    [3,3],
    [6,5],
    [7,8],
    [8,8]
])

y_train = np.array([0,0,0,1,1,1])

X_test = np.array([
    [2,2],
    [7,7]
])

k = int(input("Enter value of K: "))

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = []

        for x in X:

            # Step 1: Compute Euclidean distances
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

            # Step 2: Sort distances & get indices of K nearest
            k_indices = np.argsort(distances)[:self.k]

            # Step 3: Extract labels of K nearest points
            k_labels = self.y_train[k_indices]

            # Step 4: Majority voting
            values, counts = np.unique(k_labels, return_counts=True)
            prediction = values[np.argmax(counts)]

            predictions.append(prediction)

        return np.array(predictions)

model = KNN(k)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nPredictions:", predictions)
