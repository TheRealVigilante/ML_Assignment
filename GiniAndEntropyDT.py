import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Sample dataset: features and class labels (You can replace this with your own dataset)
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

criterion = ('gini')  # change this to "gini" or "entropy"
clf = DecisionTreeClassifier(criterion=criterion)

# Train the decision tree
clf.fit(X_encoded, y)

# Plot the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=feature_names, class_names=['Positive', 'Negative'], filled=True)
plt.show()
