# utils.py

import numpy as np
import seaborn as sns
import argparse
import yaml

# Helper functions

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process settings from a YAML file.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    return parser.parse_args()

def read_settings(config_path):
    with open(config_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings

def apply_palette(mask):
    # Define a palette
    palette = (255 * np.array(sns.color_palette('husl', 34))).astype(np.uint8)
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    for label in range(34):  
        color_mask[mask == label] = palette[label]
    
    return color_mask

def f1_score_per_class(y_true, y_pred, num_classes):
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    
    for cls in range(num_classes):
        true_positive = np.sum((y_pred == cls) & (y_true == cls))
        false_positive = np.sum((y_pred == cls) & (y_true != cls))
        false_negative = np.sum((y_pred != cls) & (y_true == cls))
        
        # Check we don't divide by zero
        if true_positive + false_positive > 0:
            precision[cls] = true_positive / (true_positive + false_positive)
        else:
            precision[cls] = 0 

        if true_positive + false_negative > 0:
            recall[cls] = true_positive / (true_positive + false_negative)
        else:
            recall[cls] = 0  

    # Calculate F1 score for each class
    f1_scores = np.zeros(num_classes)
    for i in range(num_classes):
        if (precision[i] + recall[i]) > 0:
            f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1_scores[i] = 0
    
    # Return both the mean F1 score across all classes and the individual class F1 scores
    return np.mean(f1_scores), f1_scores

def format_f1_scores(avg_f1_scores_per_class):
    formatted_scores = []
    for i, score in enumerate(avg_f1_scores_per_class):
        formatted_scores.append(f"Class {i}: {score:.4f}")
    # Join the individual class scores into a single string
    return "\n".join(formatted_scores)
