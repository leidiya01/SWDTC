import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

# Set font settings for matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Load the original Excel file
data = pd.read_excel(r'C:\Users\leidy\Desktop\clustered_data_results.xlsx')

# Define feature and target columns
dimensions = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']
target = 'Y'

# Define different sliding window sizes
window_sizes = [2, 3, 4]

# Initialize storage structures
results = []  # Store results for all window sizes and locations
training_data_dict = {}  # Store training datasets for all locations and window sizes
window_accuracies = {ws: [] for ws in window_sizes}  # Store average accuracy for each window size
window_precisions = {ws: [] for ws in window_sizes}  # Store average precision for each window size
window_recalls = {ws: [] for ws in window_sizes}  # Store average recall for each window size
window_f1_scores = {ws: [] for ws in window_sizes}  # Store average F1 scores for each window size

# Iterate through each location
for location in data['Location'].unique():
    location_data = data[data['Location'] == location].reset_index(drop=True)

    # Process each sliding window size
    for window_size in window_sizes:
        # Shuffle and split data into training and testing sets
        all_data_shuffled = shuffle(data).reset_index(drop=True)
        split_index = int(len(all_data_shuffled) * 0.7)
        X_train, y_train = all_data_shuffled[dimensions].iloc[:split_index], all_data_shuffled[target].iloc[:split_index]
        X_test, y_test = all_data_shuffled[dimensions].iloc[split_index:], all_data_shuffled[target].iloc[split_index:]

        # Save training dataset into a dictionary with a unique key
        sheet_name = f"Location_{location}_Window_{window_size}"
        training_data_dict[sheet_name] = pd.concat([X_train, y_train.rename(target)], axis=1)

        # Initialize and train the decision tree classifier
        clf = DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=None, max_leaf_nodes=4)
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred = clf.predict(X_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Save results for this location and window size
        results.append({
            "Location": location,
            "Window Size": window_size,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            **{f"Importance_{dim}": imp for dim, imp in zip(dimensions, clf.feature_importances_)}
        })

        # Add metrics to window summary
        window_accuracies[window_size].append(accuracy)
        window_precisions[window_size].append(precision)
        window_recalls[window_size].append(recall)
        window_f1_scores[window_size].append(f1)

# Calculate average metrics across all locations for each window size
average_results = []
for ws in window_sizes:
    average_results.append({
        "Location": "Average Across Locations",
        "Window Size": ws,
        "Accuracy": np.mean(window_accuracies[ws]),
        "Precision": np.mean(window_precisions[ws]),
        "Recall": np.mean(window_recalls[ws]),
        "F1 Score": np.mean(window_f1_scores[ws])
    })

# Combine results into a DataFrame
results_df = pd.DataFrame(results + average_results)

# Save all training datasets into a single Excel file
training_data_path = r'C:\Users\leidy\Desktop\training_datasets_combined.xlsx'
with pd.ExcelWriter(training_data_path) as writer:
    for sheet_name, df in training_data_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save all results to a separate Excel file
output_path = r'C:\Users\leidy\Desktop\decision_tree_classification_results.xlsx'
results_df.to_excel(output_path, index=False)

# Print paths for the saved files
print(f"Training datasets saved at: {training_data_path}")
print(f"Results saved at: {output_path}")
