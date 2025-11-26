import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

st.set_page_config(page_title="Decision Tree Hyperparameter Tuning", layout="centered")

st.title("🌿 Decision Tree Hyperparameter Tuning (Iris Dataset)")
st.write("Interactively try GridSearchCV and RandomizedSearchCV!")

# Load dataset
X, y = load_iris(return_X_y=True)

# Sidebar hyperparameters
st.sidebar.header("🔧 Hyperparameters")

max_depth_values = st.sidebar.multiselect(
    "Select values for max_depth",
    [2, 3, 4, 5, 6, 7, 8, 9, 10],
    default=[3, 5, 7]
)

min_split_values = st.sidebar.multiselect(
    "Select values for min_samples_split",
    [2, 3, 4, 5, 7, 10],
    default=[2, 4]
)

st.sidebar.write("---")

run_grid = st.sidebar.button("Run GridSearchCV")
run_random = st.sidebar.button("Run RandomizedSearchCV")

# Model
dt = DecisionTreeClassifier()

# GridSearchCV
if run_grid:
    st.subheader("📌 GridSearchCV Results")

    param_grid = {
        'max_depth': max_depth_values,
        'min_samples_split': min_split_values
    }

    grid = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)

    st.success(f"Best Params: {grid.best_params_}")
    st.write(f"Best Accuracy: {grid.best_score_}")

    # Show all combinations
    st.write("All combinations tested:")
    st.dataframe(pd.DataFrame(list(ParameterGrid(param_grid))))

# RandomizedSearchCV
if run_random:
    st.subheader("🎲 RandomizedSearchCV Results")

    param_dist = {
        'max_depth': max_depth_values,
        'min_samples_split': min_split_values
    }

    random_search = RandomizedSearchCV(dt, param_dist, n_iter=3, scoring='accuracy')
    random_search.fit(X, y)

    st.success(f"Best Params: {random_search.best_params_}")
    st.write(f"Best Accuracy: {random_search.best_score_}")
