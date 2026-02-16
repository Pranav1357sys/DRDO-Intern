import numpy as np

X = np.array([
    [1,2],
    [2,3],
    [3,3],
    [6,5],
    [7,8],
    [8,8]
])

y = np.array([0,0,0,1,1,1])

X_test = np.array([[2,2],[7,7]])

class NaiveBayes:

    def fit(self,X,y):

        self.classes = np.unique(y)
        self.mean = {}
        self.var = {}
        self.prior = {}

        for c in self.classes:
            Xc = X[y==c]
            self.mean[c] = Xc.mean(axis=0)
            self.var[c] = Xc.var(axis=0)
            self.prior[c] = Xc.shape[0]/X.shape[0]

    def gaussian(self,x,mean,var):

        num = np.exp(-(x-mean)**2/(2*var))
        den = np.sqrt(2*np.pi*var)

        return num/den

    def predict(self,X):

        preds = []

        for x in X:

            posteriors = []

            for c in self.classes:

                prior = np.log(self.prior[c])
                likelihood = np.sum(np.log(self.gaussian(x,self.mean[c],self.var[c])))

                posterior = prior + likelihood
                posteriors.append(posterior)

            preds.append(self.classes[np.argmax(posteriors)])

        return np.array(preds)

model = NaiveBayes()
model.fit(X,y)

print(model.predict(X_test))