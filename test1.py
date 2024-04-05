from io import StringIO
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import manhattan_distances

color1='#8b0000' 
color2='#7fff00'
color3='#8b0000'


def plot_confusion_matrix_similarity(cms, models):
  # Flatten each confusion matrix and calculate pairwise Manhattan distances
  flattened_cms = [cm.flatten() for cm in cms]
  distances = manhattan_distances(flattened_cms)
  
  # Apply MDS and normalize the output
  mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
  cms_2d = mds.fit_transform(distances)
  scaler = MinMaxScaler()
  cms_2d_normalized = scaler.fit_transform(cms_2d)
  
  # Plot using bubbles
  plt.figure(figsize=(10, 10))
  for i in range(len(cms_2d_normalized)):
    if i == 0:
      color = '#8b0000'
    elif i == 1:
      color = '#7fff00'
    elif i == 2:
      color = '#ff1493'
    else:
      color = 'blue'  # Add more colors if needed
    plt.scatter(cms_2d_normalized[i, 0], cms_2d_normalized[i, 1], color=color, s=2000, alpha=0.6)
    plt.text(cms_2d_normalized[i, 0], cms_2d_normalized[i, 1], models[i], ha='center', va='center')
  plt.xlabel('MDS Dimension 1 (Normalized)')
  plt.ylabel('MDS Dimension 2 (Normalized)')
  plt.title('Confusion Matrix Similarity')
  plt.grid(True)
  plt.show()

def plot_model_similarity_std_recall_accuracy(std_dev_recalls, accuracies):
  plt.figure(figsize=(10, 6))
  colors = plt.cm.plasma(np.linspace(0, 1, len(accuracies)))
  for i, color in enumerate(colors):
    plt.scatter(std_dev_recalls[i], accuracies[i], color=color, s=1000, alpha=0.7, edgecolors='black')
  plt.xlabel('Standard Deviation of Recalls')
  plt.ylabel('Accuracy')
  plt.title('Model Similarity Based on Accuracy and Std Dev of Recalls')
  plt.grid(True)
  plt.show()


def plot_error_by_class_parallel(errors_by_class, class_labels, models):
  df_errors = pd.DataFrame(errors_by_class, columns=class_labels)
  df_errors['Model'] = models
  
  plt.figure(figsize=(12, 6))
  parallel_coordinates(df_errors, 'Model', colormap=plt.cm.plasma, alpha=1)
  plt.ylabel('Error')
  plt.title('Error by Class for Each Model')
  plt.xticks(rotation=45)
  plt.show()
def std_deviation_of_recalls(confusion_matrix):
    """
    Calculate the standard deviation of recall values from a confusion matrix.
    
    Parameters:
    - confusion_matrix: 2D array-like, shape = [n_classes, n_classes]
      Confusion matrix of the model's predictions.
      
    Returns:
    - std_dev_recall: float
      The standard deviation of the recall values across all classes.
    """
    # Ensure the confusion matrix is a NumPy array for easy calculations
    cm = np.array(confusion_matrix)
    
    # Calculate recall for each class
    # Recall is defined as the true positives / (true positives + false negatives)
    recall_values = np.diag(cm) / np.sum(cm, axis=1)
    
    # Calculate and return the standard deviation of recall values
    std_dev_recall = np.std(recall_values)
    return std_dev_recall

def read_confusion_matrices(file_path):
  """
  Read a CSV file containing three confusion matrices with 11 classes.
  
  Parameters:
  - file_path: str
    The path to the CSV file.
    
  Returns:
  - cms: list
    A list of three confusion matrices.
  """
  # Read the CSV file into a pandas DataFrame
  df = pd.read_csv(file_path)
  
  # Extract the confusion matrices from the DataFrame
  cms = []
  for i in range(11):
    cm = df.iloc[i:, i:].values.astype(int)
    cms.append(cm)
  
  return cms
import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics_and_labels(confusion_matrices, models):
  """Calculate precision, recall, and F1-score for each confusion matrix and label them."""
  metrics = []
  
  for i, cm in enumerate(confusion_matrices):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    model_label = models[i]  # Use the model name from the models list
    
    for score in precision[~np.isnan(precision)]:
      metrics.append({'Metric': 'Precision', 'Score': score, 'Model': model_label})
    for score in recall[~np.isnan(recall)]:
      metrics.append({'Metric': 'Recall', 'Score': score, 'Model': model_label})
    for score in f1_score[~np.isnan(f1_score)]:
      metrics.append({'Metric': 'F1-Score', 'Score': score, 'Model': model_label})
  
  return pd.DataFrame(metrics)

def plot_metrics_with_labels(metrics_df):
    """Plot box plots for precision, recall, and F1-score with model labels."""
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Metric', y='Score', hue='Model', data=metrics_df, palette='Set1')
    plt.title('Distribution of Precision, Recall, and F1-Score Across Models')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.show()

# Assuming matrices_list is your list of confusion matrices
# Note: Replace the dummy extraction with your actual matrices extraction method





def calculate_errors_by_class(confusion_matrices):
    """
    Calculate the errors by class for each confusion matrix.
    
    Parameters:
    - confusion_matrices: list of np.array
      A list of confusion matrices, each as a NumPy array.
      
    Returns:
    - errors_by_class: list of np.array
      A list where each element is an array of error rates for each class in the corresponding confusion matrix.
    """
    errors_by_class = []
    for cm in confusion_matrices:
        # Calculate recall for each class. Recall = TP / (TP + FN)
        recall_per_class = np.diag(cm) / np.sum(cm, axis=1)
        # Calculate error for each class. Error = 1 - Recall
        error_per_class = 1 - recall_per_class
        errors_by_class.append(error_per_class)
    
    return errors_by_class

# Example usage:

# Assuming matrices is the list of three confusion matrices you provided
#matrices_list = extract_confusion_matrices(csv_data)  # Using the function from the previous step to extract matrices
#errors_list = calculate_errors_by_class(matrices_list)

# Printing errors by class for each confusion matrix
#for i, errors in enumerate(errors_list):
#    print(f"Errors for Matrix {i+1}: {errors}\n")

# Function to read the CSV data and return a list of three matrices contained
def extract_confusion_matrices(csv_data):
    
  # Load the CSV data into a pandas DataFrame
  df = pd.read_csv(StringIO(csv_data))
  rows_per_matrix = 11  # Assuming each matrix is 11x11
  num_matrices = len(df) // rows_per_matrix
  
  matrices = []
  for i in range(num_matrices):
    start_row = i * rows_per_matrix
    end_row = start_row + rows_per_matrix
    matrix = df.iloc[start_row:end_row].values
    matrices.append(matrix)
    
  return matrices

if __name__ == "__main__":


  file_path = "C:/Users/edgar/Documents/MLOPS/cm-results1.csv"
  csv_data = """
  class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,class11
  141,7,0,0,0,0,0,0,0,2,0
  1,84,0,0,0,0,0,0,0,0,0
  0,0,6,0,0,0,0,0,0,0,0
  0,0,0,37,0,0,0,0,0,0,0
  0,0,0,0,15,0,0,0,0,0,0
  0,0,0,0,0,20,0,0,0,0,0
  0,0,0,0,0,0,14,0,0,0,0
  0,0,0,0,0,0,0,59,0,0,0
  0,0,0,0,0,0,0,0,39,0,4
  1,0,0,0,0,0,0,0,0,153,0
  0,0,0,0,0,0,0,0,0,0,20
  126,4,0,0,0,0,0,0,0,3,0
  6,84,0,0,0,0,0,0,0,0,0
  0,0,11,0,0,0,0,0,0,0,0
  0,0,0,36,0,0,0,0,0,0,0
  0,0,0,0,12,0,0,0,0,0,0
  0,0,0,1,2,23,0,0,0,0,0
  0,0,0,0,0,4,13,0,0,0,0
  0,0,0,0,0,0,0,58,0,0,0
  0,0,0,0,0,0,0,0,43,0,0
  3,0,0,0,0,0,0,0,0,149,0
  0,0,0,0,0,0,0,0,0,0,25
  110,15,0,0,0,0,0,0,0,0,0
  0,95,0,0,0,0,0,0,0,0,0
  0,0,5,0,0,0,0,0,0,0,0
  0,1,0,29,0,0,0,0,0,1,0
  0,0,0,0,22,0,0,0,0,0,0
  0,0,0,0,0,17,0,0,0,0,0
  0,0,0,0,0,0,12,0,0,0,0
  0,0,0,0,0,0,0,46,0,0,0
  0,0,0,0,0,0,0,0,56,0,0
  4,1,0,0,0,0,0,0,0,161,0
  0,0,0,0,0,0,0,2,1,0,25
  """
  matrices=extract_confusion_matrices(csv_data)
  models=["convnextv2_tiny", "deit3_base_patch16_224", "swin_tiny_patch4_window7_224"]
  plot_confusion_matrix_similarity(matrices, models)
  calculate_errors_by_class(matrices)
# Example usage of the function

  for i, matrix in enumerate(matrices):
      print(f"Matrix {i+1}:\n", matrix, "\n")

  class_labels=["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"]
  errors_by_class = calculate_errors_by_class(matrices)
  plot_error_by_class_parallel(errors_by_class, class_labels, models)
  
  metrics_df = calculate_metrics_and_labels(matrices,models)
  plot_metrics_with_labels(metrics_df)


