import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("winequality-red.csv")

# Convert the numeric quality score into a categorical variable:
#   quality <= 5 -> 'low'
#   quality == 6 -> 'medium'
#   quality >= 7 -> 'high'
def quality_to_category(q):
    if q <= 5:
        return 'low'
    elif q == 6:
        return 'medium'
    else:
        return 'high'

df["quality_cat"] = df["quality"].apply(quality_to_category)

# Select a subset of quantitative features (here we choose 5 features)
features = ['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'alcohol', 'fixed acidity', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH','sulphates']

# Define predictors (X) and target (y)
X = df[features]
y = df["quality_cat"]

X -= np.average(X, axis=0)
X /= np.std(X, axis=0)

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ------------- GRID SEARCH FOR SVC -------------

clf = SVC(kernel='rbf')
parameters = {"C": np.logspace(-2, 3, 6), #np.linspace(8, 20, 20),
              "gamma": np.logspace(-4, 0, 5)} #np.linspace(0.001, .01, 10)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_C', 'param_gamma', 'mean_test_score', 'rank_test_score']])
print(grid_search.best_params_)

# Train SVC with the best parameters on full training data
svc_best = SVC(C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'],kernel='rbf')
svc_best.fit(X_train, y_train)

print(f"SVC Train Score: {svc_best.score(X_train, y_train):.3f}")
print(f"SVC Test Score: {svc_best.score(X_test, y_test):.3f}")

# Plot the confusion matrix for SVC
cm_svc = confusion_matrix(y_test, svc_best.predict(X_test), normalize="true")
disp_svc = ConfusionMatrixDisplay(cm_svc, display_labels=svc_best.classes_)
disp_svc.plot()
plt.title("SVC Confusion Matrix")
plt.show()


#
# # ------------- GRID SEARCH FOR RANDOM FOREST -------------
parameters_rf = {
    "max_depth": range(2, 16)
}

# Set up the Random Forest classifier and grid search
clf_rf = RandomForestClassifier()
grid_search_rf = GridSearchCV(clf_rf, param_grid=parameters_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Convert grid search results to a DataFrame and print select columns
results_rf = pd.DataFrame(grid_search_rf.cv_results_)
print(results_rf[[ 'param_max_depth', 'mean_test_score', 'rank_test_score']])
print("Best Random Forest Parameters:", grid_search_rf.best_params_)


rfc = RandomForestClassifier(max_depth=grid_search_rf.best_params_['max_depth'], oob_score=True, verbose=3, n_jobs=-1)
rfc.fit(X_train, y_train)
# Train the Random Forest classifier with the best parameters on the full training data


print(f"Random Forest Train Score: {rfc.score(X_train, y_train):.3f}")
print(f"Random Forest Test Score: {rfc.score(X_test, y_test):.3f}")

# Plot the normalized confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, rfc.predict(X_test), normalize="true")
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rfc.classes_)
disp_rf.plot()
plt.title("Random Forest Confusion Matrix")
plt.show()