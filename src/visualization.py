"""Reusable chart helpers for the Streamlit application."""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def plot_feature_importance(names, values, top_n=12):
    pairs = sorted(zip(names, values), key=lambda item: item[1], reverse=True)[:top_n]
    labels, scores = zip(*pairs)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(scores), y=list(labels), ax=ax, color="#38bdf8")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig
