# streamlit_adaboost_pro_compact_dashboard.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AdaBoost Compact Dashboard", layout="wide")
st.title("AdaBoost Compact Dashboard")

# ======================
# Dataset Controls
# ======================
st.sidebar.header("Dataset & Model Controls")
num_points = st.sidebar.slider("Number of points", 5, 20, 5)
num_learners = st.sidebar.slider("Number of weak learners", 1, 5, 3)
feature_choice = st.sidebar.selectbox("Choose feature for threshold", ["Auto", "X1", "X2"])
manual_threshold = st.sidebar.slider("Manual threshold (if selected)", 0.0, 6.0, 3.0)

if "X" not in st.session_state or st.sidebar.button("Generate New Dataset"):
    np.random.seed(42)
    st.session_state.X = np.random.rand(num_points, 2) * 6
    st.session_state.y = np.random.choice([1, -1], size=num_points)

X = st.session_state.X
y = st.session_state.y

# ======================
# Helper Functions
# ======================
def train_adaboost(X, y, num_learners, feature_choice="Auto", manual_threshold=None):
    w = np.ones(len(y)) / len(y)
    alphas, weak_learners = [], []
    thresholds_x1, thresholds_x2 = np.unique(X[:,0]), np.unique(X[:,1])
    weight_history, error_history, alpha_history = [], [], []

    for t in range(num_learners):
        best_error = float('inf')
        feature_list = [0,1] if feature_choice=="Auto" else ([0] if feature_choice=="X1" else [1])
        thresholds = thresholds_x1 if feature_list[0]==0 else thresholds_x2

        if manual_threshold is not None and feature_choice != "Auto":
            thresholds = [manual_threshold]

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

        w = w * np.exp(-alpha * y * best_pred)
        w /= np.sum(w)

        weight_history.append(w.copy())
        error_history.append(best_error)
        alpha_history.append(alpha)

    return alphas, weak_learners, weight_history, error_history, alpha_history

def strong_classifier(X_input, alphas, weak_learners):
    final_pred = np.zeros(X_input.shape[0])
    for alpha, (feature, thresh, _) in zip(alphas, weak_learners):
        final_pred += alpha * np.where(X_input[:,feature] < thresh, 1, -1)
    return np.sign(final_pred)

# ======================
# Train AdaBoost
# ======================
alphas, weak_learners, weight_history, error_history, alpha_history = train_adaboost(
    X, y, num_learners, feature_choice, manual_threshold
)
y_pred_final = strong_classifier(X, alphas, weak_learners)
st.write("**Final Predictions:**", y_pred_final)

# ======================
# Compact Mini-Dashboard
# ======================
st.subheader("Mini Dashboard: Weights & Decision Boundaries")

xx, yy = np.meshgrid(np.linspace(0,6,200), np.linspace(0,6,200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Create columns for each weak learner
cols = st.columns(num_learners)

for t, col in enumerate(cols):
    with col:
        # ----- Weight Plot -----
        fig_w, ax_w = plt.subplots(figsize=(1.2,1))
        colors = [np.array([0,0,1])*weight_history[t][i] + 0.5*(1-weight_history[t][i]) if y[i]==1
                  else np.array([1,0,0])*weight_history[t][i] + 0.5*(1-weight_history[t][i]) for i in range(len(y))]
        ax_w.scatter(X[:,0], X[:,1], s=weight_history[t]*200+20, color=colors, edgecolor='black')
        mis_idx = np.where(weak_learners[t][2]!=y)[0]
        ax_w.scatter(X[mis_idx,0], X[mis_idx,1], facecolors='none', edgecolors='yellow',
                     s=weight_history[t][mis_idx]*300+20, linewidths=1.5)
        if weak_learners[t][0]==0:
            ax_w.axvline(x=weak_learners[t][1], color='green', linestyle='--', linewidth=1)
        else:
            ax_w.axhline(y=weak_learners[t][1], color='green', linestyle='--', linewidth=1)
        ax_w.set_xticks([])
        ax_w.set_yticks([])
        ax_w.set_title(f"W{t+1}", fontsize=8)
        col.pyplot(fig_w)

        # ----- Decision Boundary Plot -----
        final_pred_partial = np.zeros(grid.shape[0])
        for alpha, (feature, thresh, _) in zip(alphas[:t+1], weak_learners[:t+1]):
            final_pred_partial += alpha * np.where(grid[:,feature] < thresh, 1, -1)
        Z = np.sign(final_pred_partial).reshape(xx.shape)
        fig_d, ax_d = plt.subplots(figsize=(1.2,1))
        ax_d.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.bwr)
        ax_d.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', s=20)
        ax_d.scatter(X[y==-1][:,0], X[y==-1][:,1], color='red', s=20)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
        ax_d.set_title(f"D{t+1}", fontsize=8)
        col.pyplot(fig_d)
