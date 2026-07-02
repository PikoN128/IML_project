import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------
# 1. Setup
# ----------------------------
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Feature indices
FEATURES = {
    "mfcc": list(range(24)),
    "centroid": [24, 25],
    "bandwidth": [26, 27],
    "rolloff": [28, 29],
    "zcr": [30, 31],
    "rms": [32, 33]
}

# Select the feature groups to use
selected_features = (
    FEATURES["mfcc"] +
    FEATURES["centroid"] +
    FEATURES["bandwidth"] +
    FEATURES["rms"]
)

# ----------------------------
# Load features
# ----------------------------
data = np.load(
    'C:/Users/User/Desktop/MLProject/features_34.npy',
    allow_pickle=True
)

X = data.reshape(-1, data.shape[-1])

# Keep only selected features
X = X[:, selected_features]

# Labels
y = np.repeat(genres, 100)

# ----------------------------
# 2. Train / Validation / Test split
# ----------------------------
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=2/9,
    stratify=y_dev,
    random_state=42
)

# ----------------------------
# 3. Standardization
# ----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ----------------------------
# 4. Hyperparameter sweep
# ----------------------------
metrics = [
    'euclidean',
    'manhattan',
    'cosine',
    'chebyshev',
    'minkowski'
]

k_values = range(1, 16, 2)

results = []

best_score = -1
best_k = None
best_metric = None

print("\nValidation results")
print("-" * 50)

for metric in metrics:
    for k in k_values:

        knn = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric
        )

        knn.fit(X_train, y_train)

        score = knn.score(X_val, y_val)

        results.append({
            "metric": metric,
            "k": k,
            "accuracy": score
        })

        print(f"{metric:10s}  k={k:2d}  val_acc={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_metric = metric

print("\nBest model")
print(f"Metric : {best_metric}")
print(f"k      : {best_k}")
print(f"Val acc: {best_score:.4f}")

# ----------------------------
# 5. Plot validation accuracy
# ----------------------------
plt.figure(figsize=(8, 5))

for metric in metrics:

    metric_results = [r for r in results if r["metric"] == metric]

    ks = [r["k"] for r in metric_results]
    accs = [r["accuracy"] for r in metric_results]

    plt.plot(
        ks,
        accs,
        marker='o',
        label=metric
    )

plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title(f"kNN Hyperparameter Search ({len(selected_features)} Features)")
plt.xticks(list(k_values))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 6. Retrain on train + validation
# ----------------------------
final_scaler = StandardScaler()

X_final = final_scaler.fit_transform(X_dev)
X_test_final = final_scaler.transform(X_test)

final_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric=best_metric
)

final_knn.fit(X_final, y_dev)

# ----------------------------
# 7. Final test evaluation
# ----------------------------
test_accuracy = final_knn.score(X_test_final, y_test)

print(f"\nFinal test accuracy: {test_accuracy:.4f}")

# ----------------------------
# 8. Confusion matrix
# ----------------------------
y_pred = final_knn.predict(X_test_final)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=genres
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=genres
)

fig, ax = plt.subplots(figsize=(8, 8))

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)

plt.title(f"Confusion Matrix ({len(selected_features)} Features)")
plt.tight_layout()
plt.show()