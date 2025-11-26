from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.tree import DecisionTreeClassifier

df = load_iris()

#Print df
print("\n",df)

#Load dataset
X, y = load_iris(return_X_y = True)
print(y)

#Model
dt = DecisionTreeClassifier()

#Define hyperparameter
param_grid = {
    'max_depth':[3,7,5,9],
    'min_samples_split' :[2,4]
}

#Generate all combinations
grid_table = list(ParameterGrid(param_grid))
print("\nAll possible comninations: ")
print(grid_table)

grid = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid.fit(X,y)
print(f"\nBest Combination (GridSearch): {grid.best_params_}")
print(f"\nBest Score: {grid.best_score_}")

#RandomizedSearchCV
param_dist = {
    'max_depth': [2,4,6,8,10],
    'min_samples_split': [3,5,7]
}

random_search = RandomizedSearchCV(dt, param_dist, n_iter=3, scoring='accuracy')
random_search.fit(X,y)
print(f"\nBest Combinations (RandomizedSearch): {random_search.best_params_}")
print(f"\nBest Accuracy: {random_search.best_score_}")