import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Sample dataset: features and class labels
data = {
    'Color': ["Red", "Blue", "Green", "Red", "Green", "Green", "Blue", "Blue", "Red", "Blue", "Green", "Red", "Green", "Green"],
    'Type': ["SUV", "Minivan", "Car", "Minivan", "Car", "SUV", "SUV", "Car", "SUV", "Car", "SUV", "Car", "SUV", "Minivan"],
    'Doors': [2, 4, 4, 4, 2, 4, 2, 2, 2, 4, 4, 2, 2, 4],
    'Tires': ["Whitewall", "Whitewall", "Whitewall", "Blackwall", "Blackwall", "Blackwall", "Blackwall", "Whitewall", "Blackwall", "Blackwall", "Whitewall", "Blackwall", "Blackwall", "Whitewall"],
    'Class': ['Positive', 'Negative', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Negative', 'Negative', 'Positive', 'Positive', 'Negative', 'Negative']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Color', 'Type', 'Doors', 'Tires']]  # Independent variables
y = df['Class']  # Target variable (positive/negative)

# Apply One-Hot Encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Get the feature names after encoding
feature_names = encoder.get_feature_names_out(X.columns)

# Initialize variables for tracking the best model
best_misclassification_rate = float('inf')
best_classifier = None
best_params = {}

# Try different configurations
for criterion in ['gini', 'entropy', 'log_loss']:
    for max_depth in [None, 1, 2, 3, 4, 5]:
        for min_samples_leaf in [1, 2, 3, 4, 5]:
            # Initialize the classifier with current parameters
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

            # Train the decision tree
            clf.fit(X_encoded, y)

            # Make predictions
            y_pred = clf.predict(X_encoded)

            # Calculate accuracy and misclassification rate
            accuracy = accuracy_score(y, y_pred)
            misclassification_rate = 1 - accuracy

            # Check if this is the best model so far
            if misclassification_rate < best_misclassification_rate:
                best_misclassification_rate = misclassification_rate
                best_classifier = clf
                best_params = {
                    'Criterion': criterion,
                    'Max Depth': max_depth,
                    'Min Samples Leaf': min_samples_leaf
                }

# Print the best parameters and misclassification rate
print(f"Best Misclassification Rate: {best_misclassification_rate:.2f}")
print(f"Best Parameters: {best_params}")

# Plot the best decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(best_classifier, feature_names=feature_names, class_names=['Positive', 'Negative'], filled=True)
plt.title("Best Decision Tree")
plt.show()
