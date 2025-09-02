# AdaBoost from Scratch with 2D Visualization
import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1️⃣ Dataset
# ======================
X = np.array([[1, 2], [2, 1], [3, 3], [4, 2], [5, 1]])
y = np.array([1, 1, -1, -1, -1])

# Initial dataset plot
plt.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='+1')
plt.scatter(X[y==-1][:,0], X[y==-1][:,1], color='red', label='-1')
plt.title("Initial Dataset")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()

# ======================
# 2️⃣ AdaBoost Implementation
# ======================
# Initialize weights
w = np.ones(len(y)) / len(y)
alphas = []
weak_learners = []

# Candidate thresholds for decision stumps
thresholds_x1 = np.unique(X[:,0])
thresholds_x2 = np.unique(X[:,1])

# Number of weak learners
num_learners = 3

for t in range(num_learners):
    best_error = float('inf')
    
    # Try splitting on x1
    for thresh in thresholds_x1:
        pred = np.where(X[:,0] < thresh, 1, -1)
        error = np.sum(w * (pred != y))
        if error < best_error:
            best_error = error
            best_pred = pred
            best_feature = 0
            best_thresh = thresh
    
    # Try splitting on x2
    for thresh in thresholds_x2:
        pred = np.where(X[:,1] < thresh, 1, -1)
        error = np.sum(w * (pred != y))
        if error < best_error:
            best_error = error
            best_pred = pred
            best_feature = 1
            best_thresh = thresh
    
    # Compute alpha
    alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
    alphas.append(alpha)
    weak_learners.append((best_feature, best_thresh, best_pred))
    
    # Update weights
    w = w * np.exp(-alpha * y * best_pred)
    w /= np.sum(w)
    
    # Visualization of weights
    plt.figure()
    plt.scatter(X[:,0], X[:,1], s=w*500, c=['blue' if label==1 else 'red' for label in y])
    plt.title(f"Weights after weak learner {t+1} (feature {best_feature}, thresh {best_thresh})")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

# ======================
# 3️⃣ Strong Classifier
# ======================
def strong_classifier(X_input):
    final_pred = np.zeros(X_input.shape[0])
    for alpha, (feature, thresh, _) in zip(alphas, weak_learners):
        pred = np.where(X_input[:,feature] < thresh, 1, -1)
        final_pred += alpha * pred
    return np.sign(final_pred)

# Predictions on training data
y_pred = strong_classifier(X)
print("Final predictions on training data:", y_pred)

# ======================
# 4️⃣ Decision Boundary Visualization
# ======================
xx, yy = np.meshgrid(np.linspace(0,6,200), np.linspace(0,4,200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = strong_classifier(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)
plt.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='+1')
plt.scatter(X[y==-1][:,0], X[y==-1][:,1], color='red', label='-1')
plt.title("AdaBoost Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
