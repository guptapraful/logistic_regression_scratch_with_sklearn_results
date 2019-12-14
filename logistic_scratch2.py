import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

# plt.figure(figsize=(12,8))
# plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
#             c = simulated_labels, alpha = .4)


####### SCRATCH IMPLEMENTATION OF LOGISTIC REGRESSION
class Logistic_regression_custom_reg():

    def __init__(self, num_steps, learning_rate=1e-2, C = 1.0, add_intercept = False):
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.C = C
        self.add_intercept = add_intercept

    def sigmoid(self, scores):
        return 1 / (1 + np.exp(-scores))

    def reg_logLiklihood(self, features, target):
        scores = np.dot(features, self.weights)
        reg_term = (1 / (2 * self.C)) * np.dot(self.weights.T, self.weights)
        return -1 * np.sum((target * np.log(self.sigmoid(scores))) + ((1 - target) * np.log(1 - self.sigmoid(scores)))) + reg_term


    def fit(self, features, target):
        if self.add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
            
        self.weights = np.zeros(features.shape[1])
        
        for step in xrange(self.num_steps):
            scores = np.dot(features, self.weights)
            predictions = self.sigmoid(scores)

            output_error_signal = target - predictions
            gradient = self.C*np.dot(features.T, output_error_signal) + np.sum(self.weights)
            delta_w = self.learning_rate * gradient
            self.weights += delta_w
            
            if step % 10000 == 0:
                print self.reg_logLiklihood(features, target)

    def predict_proba(self, features):
        z = self.weights[0] + np.dot(features, self.weights[1:])        
        probs = np.array([self.sigmoid(i) for i in z])        
        return probs


model = Logistic_regression_custom_reg(num_steps = 300000, learning_rate = 7e-7, C = 1.0, add_intercept=True)
model.fit(simulated_separableish_features, simulated_labels)
scratch_probs = model.predict_proba(simulated_separableish_features)

####### SKLEARN RESULTS
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 1.0)
clf.fit(simulated_separableish_features, simulated_labels)
sklearn_probs = clf.predict_proba(simulated_separableish_features)[:,1]

plt.figure()
plt.scatter(sklearn_probs, scratch_probs)
plt.show()

# print clf.intercept_, clf.coef_
# print probs

