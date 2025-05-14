import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


# https://www.kaggle.com/datasets/oktayrdeki/heart-disease
df = pd.read_csv("car_evaluation.csv")

df=df.rename(columns = {"vhigh" : "buying_price"})
df=df.rename(columns = {"vhigh.1" : "maintenance_cost"})
df=df.rename(columns = {"2" : "num_doors"})
df=df.rename(columns = {"2.1" : "num_people"})
df=df.rename(columns = {"small" : "storage"})
df=df.rename(columns = {"low" : "safety"})
df=df.rename(columns = {"unacc" : "decision"})

cat_columns = df.select_dtypes("object").columns

df[cat_columns] = df[cat_columns].astype("category")
cat_dict = {cat_columns[i]: {j: df[cat_columns[i]].cat.categories[j] for j in
                             range(len(df[cat_columns[i]].cat.categories))} for i in range(len(cat_columns))}
print(cat_dict)
df[df.select_dtypes("category").columns] = df[df.select_dtypes("category").columns].apply(lambda x: x.cat.codes)
df.dropna(inplace=True)

X = df.iloc[:, :-1].copy().to_numpy()
y = df.iloc[:, -1].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier(class_weight='balanced')
parameters = {"max_depth": range(2, 20)}
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)
print(results[['param_max_depth', 'mean_test_score', 'rank_test_score']])

clf = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth'], class_weight='balanced')
clf.fit(X_train, y_train)

print(f"Test Score: {clf.score(X_test, y_test):.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test))
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()


plt.figure(figsize=(20, 10))

plot_tree(clf, feature_names=list(df.columns[:-1]),
          class_names=[str(c) for c in clf.classes_],
          filled=True, max_depth=3, rounded=True)
plt.title("Car Evaluation Decision Tree (Top 3 Levels)")
plt.show()



