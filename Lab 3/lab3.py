import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
data = load_wine()  # You can replace this with another dataset
X, y = data.data, data.target

# Split data for training and testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the classifiers
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="linear")
}

# Perform 10-fold cross-validation and store results
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = {}

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
    cv_results[name] = np.mean(scores)
    print(f"{name} - Mean Accuracy: {np.mean(scores):.4f}")

# Find the best model based on highest cross-validation score
best_model_name = max(cv_results, key=cv_results.get)
best_model = classifiers[best_model_name]

print(f"\nBest Model: {best_model_name} with Accuracy {cv_results[best_model_name]:.4f}")

# Train the best model on the training data and evaluate on the test data
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.show()
