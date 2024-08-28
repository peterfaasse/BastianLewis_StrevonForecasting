import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# Plotting the confusion matrix
def plot_confusion_matrix(cm, class_names, filename='confusion_matrix.png', dpi=300):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(filename, dpi=dpi)
    plt.show()


def save_classification_report_as_image(cr, filename='classification_report.png', figsize=(10, 5), dpi=300):
    """
    Save the classification report as an image.

    Parameters:
    - cr: dict, classification report generated by sklearn's classification_report with output_dict=True.
    - filename: str, name of the output image file (default is 'classification_report.png').
    - figsize: tuple, size of the figure (default is (10, 5)).
    - dpi: int, resolution of the output image (default is 300).
    """
    # Convert the classification report dictionary to a pandas DataFrame
    report_df = pd.DataFrame(cr).transpose()

    # Plot the table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=report_df.values, colLabels=report_df.columns, rowLabels=report_df.index, cellLoc='center', loc='center')

    # Save the table as an image
    plt.savefig(filename, bbox_inches='tight', dpi=dpi)
    plt.close()


def save_probability_histplot(y_proba_rfe, y_test, filepath, bins=20, figsize=(10, 6), dpi=300):
    """
    Creates and saves a histogram plot of predicted probabilities for two classes.

    Parameters:
    - y_proba_rfe: array-like, predicted probabilities.
    - y_test: array-like, true labels.
    - filepath: str, file path where the image will be saved.
    - bins: int, number of bins for the histogram (default is 20).
    - figsize: tuple, size of the figure (default is (10, 6)).
    - dpi: int, resolution of the output image (default is 300).
    """
    # Split the probabilities based on the true labels
    y_proba_starters = y_proba_rfe[y_test == 1]
    y_proba_non_starters = y_proba_rfe[y_test == 0]

    # Create the histogram plot
    plt.figure(figsize=figsize)

    # Histogram for starters
    plt.hist(y_proba_starters, bins=bins, alpha=0.6, color='blue', label='Starters', edgecolor='black')

    # Histogram for non-starters
    plt.hist(y_proba_non_starters, bins=bins, alpha=0.6, color='red', label='Non-Starters', edgecolor='black')

    # Add labels and title
    plt.xlabel('Predicted Probability for Starters')
    plt.ylabel('Number of Candidates')
    plt.title('Histogram of Predicted Probabilities')
    plt.legend(loc='upper right')

    # Save the plot as an image
    plt.savefig(filepath, bbox_inches='tight', dpi=dpi)
    plt.close()