import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Target")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
id3 = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)
c45 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)
cart = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)
models = {
    "ID3": id3,
    "C4.5": c45,
    "CART": cart
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted"
    )
    recall = recall_score(
        y_test, y_pred, average="weighted"
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted"
    )
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Tree Depth": model.get_depth(),
        "Nodes": model.tree_.node_count
    })
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("       DECISION TREE ALGORITHM COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))
plt.figure(figsize=(10, 6))
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
x = np.arange(len(models))
width = 0.18
for i, metric in enumerate(metrics):
    plt.bar(
        x + (i - 1.5) * width,
        results_df[metric],
        width,
        label=metric
    )
plt.xticks(x, results_df["Algorithm"])
plt.ylim(0, 1.1)
plt.title(
    "Performance Comparison of ID3, C4.5 and CART",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Decision Tree Algorithm")
plt.ylabel("Score")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
plt.figure(figsize=(9, 5))
x = np.arange(len(models))
plt.bar(
    x - 0.2,
    results_df["Tree Depth"],
    width=0.4,
    label="Tree Depth"
)
plt.bar(
    x + 0.2,
    results_df["Nodes"],
    width=0.4,
    label="Number of Nodes"
)
plt.xticks(x, results_df["Algorithm"])
plt.title(
    "Decision Tree Complexity",
    fontsize=16,
    fontweight="bold"
)
plt.xlabel("Algorithm")
plt.ylabel("Complexity")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        cbar=False,
        ax=ax
    )
    ax.set_title(
        f"{name} Confusion Matrix",
        fontweight="bold"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()
for name, model in models.items():
    plt.figure(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title(
        f"{name} Decision Tree",
        fontsize=18,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.show()
best_algorithm = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Algorithm"
]
best_accuracy = results_df["Accuracy"].max()
print("\n" + "=" * 70)
print(f"BEST PERFORMING ALGORITHM: {best_algorithm}")
print(f"BEST ACCURACY: {best_accuracy * 100:.2f}%")
print("=" * 70)
