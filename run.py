# streamlit_adaboost_pro_resized_compact.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AdaBoost Interactive Visualization", layout="wide")
st.title("AdaBoost Interactive Visualization (Compact)")

# ======================
# 1️⃣ Dataset Controls
# ======================
st.sidebar.header("Dataset & Model Controls")
num_points = st.sidebar.slider("Number of points", 5, 20, 5)
num_learners = st.sidebar.slider("Number of weak learners", 1, 5, 3)
feature_choice = st.sidebar.selectbox("Choose feature for threshold", ["Auto", "X1", "X2"])
manual_threshold = st.sidebar.slider("Manual threshold (if selected)", 0.0, 6.0, 3.0)

# Reset Dataset button
if "X" not in st.session_state or st.sidebar.button("Generate New Dataset"):
    np.random.seed(42)
    st.session_state.X = np.random.rand(num_points, 2) * 6
    st.session_state.y = np.random.choice([1, -1], size=num_points)

X = st.session_state.X
y = st.session_state.y

# ======================
# 2️⃣ Helper Functions
# ======================
def train_adaboost(X, y, num_learners, feature_choice="Auto", manual_threshold=None):
    w = np.ones(len(y)) / len(y)
    alphas = []
    weak_learners = []
    thresholds_x1 = np.unique(X[:,0])
    thresholds_x2 = np.unique(X[:,1])
    
    weight_history = []
    error_history = []
    alpha_history = []
    
    for t in range(num_learners):
        best_error = float('inf')
        
        # Decide which feature to split
        if feature_choice == "X1":
            feature_list = [0]
            thresholds = [manual_threshold] if manual_threshold else thresholds_x1
        elif feature_choice == "X2":
            feature_list = [1]
            thresholds = [manual_threshold] if manual_threshold else thresholds_x2
        else:
            feature_list = [0,1]
        
        # Find best split
        for f in feature_list:
            thresh_list = thresholds_x1 if f==0 else thresholds_x2
            for thresh in thresh_list:
                pred = np.where(X[:,f] < thresh, 1, -1)
                error = np.sum(w * (pred != y))
                if error < best_error:
                    best_error = error
                    best_pred = pred
                    best_feature = f
                    best_thresh = thresh
        
        alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
        alphas.append(alpha)
        weak_learners.append((best_feature, best_thresh, best_pred))
        
        # Update weights
        w = w * np.exp(-alpha * y * best_pred)
        w /= np.sum(w)
        
        weight_history.append(w.copy())
        error_history.append(best_error)
        alpha_history.append(alpha)
        
    return alphas, weak_learners, weight_history, error_history, alpha_history

def strong_classifier(X_input, alphas, weak_learners):
    final_pred = np.zeros(X_input.shape[0])
    for alpha, (feature, thresh, _) in zip(alphas, weak_learners):
        pred = np.where(X_input[:,feature] < thresh, 1, -1)
        final_pred += alpha * pred
    return np.sign(final_pred)

# ======================
# 3️⃣ Train AdaBoost
# ======================
alphas, weak_learners, weight_history, error_history, alpha_history = train_adaboost(
    X, y, num_learners, feature_choice, manual_threshold
)

y_pred_final = strong_classifier(X, alphas, weak_learners)
st.write("**Final Predictions:**", y_pred_final)

# ======================
# 4️⃣ Step-by-Step Weight Visualization (Compact)
# ======================
st.subheader("Step-by-Step Weight Updates")
for t in range(num_learners):
    fig, ax = plt.subplots(figsize=(2.5,2))  # ~50% of default
    colors = []
    for i, label in enumerate(y):
        base_color = np.array([0,0,1]) if label==1 else np.array([1,0,0])
        weight_scaled = weight_history[t][i]
        colors.append(base_color * weight_scaled + 0.5*(1-weight_scaled))
    ax.scatter(X[:,0], X[:,1], s=weight_history[t]*500 + 50, color=colors, edgecolor='black')
    
    mis_idx = np.where(weak_learners[t][2] != y)[0]
    ax.scatter(X[mis_idx,0], X[mis_idx,1], facecolors='none', edgecolors='yellow',
               s=weight_history[t][mis_idx]*700 + 50, linewidths=2)
    
    if weak_learners[t][0]==0:
        ax.axvline(x=weak_learners[t][1], color='green', linestyle='--')
    else:
        ax.axhline(y=weak_learners[t][1], color='green', linestyle='--')
    
    ax.set_title(f"Weights after Weak Learner {t+1} (α={alpha_history[t]:.2f}, error={error_history[t]:.2f})", fontsize=8)
    ax.set_xlabel("X1", fontsize=7)
    ax.set_ylabel("X2", fontsize=7)
    st.pyplot(fig)

# ======================
# 5️⃣ Decision Boundary (Compact)
# ======================
st.subheader("Decision Boundary Evolution")
xx, yy = np.meshgrid(np.linspace(0,6,200), np.linspace(0,6,200))
grid = np.c_[xx.ravel(), yy.ravel()]

for t in range(num_learners):
    final_pred_partial = np.zeros(grid.shape[0])
    for alpha, (feature, thresh, _) in zip(alphas[:t+1], weak_learners[:t+1]):
        pred = np.where(grid[:,feature] < thresh, 1, -1)
        final_pred_partial += alpha * pred
    Z = np.sign(final_pred_partial).reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(2.5,2))  # ~50% of default
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)
    ax.scatter(X[y==1][:,0], X[y==1][:,1], color='blue')
    ax.scatter(X[y==-1][:,0], X[y==-1][:,1], color='red')
    ax.set_title(f"Decision Boundary after Weak Learner {t+1}", fontsize=8)
    ax.set_xlabel("X1", fontsize=7)
    ax.set_ylabel("X2", fontsize=7)
    st.pyplot(fig)
