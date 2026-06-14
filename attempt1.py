import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Setup
# ----------------------------
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load features: expected shape (10, 100, 24)
data = np.load(
    'C:/Users/User/Desktop/MLProject/features_train.npy',
    allow_pickle=True
)

# Convert to (1000, 24)
X = data.reshape(-1, data.shape[-1])

# Labels: 100 samples per genre
y = np.repeat(genres, 100)

# ----------------------------
# 2. Train / Val / Test split
# ----------------------------
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.1,
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=2/9,  # gives 70/20/10 overall
    stratify=y_dev,
    random_state=42
)

# ----------------------------
# 3. Standardization (VERY IMPORTANT for kNN)
# ----------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ----------------------------
# 4. Hyperparameter sweep
# ----------------------------
metrics = ['euclidean', 'manhattan', 'cosine','hamming','chebyshev','minkowski']
k_values = range(1, 16, 2)

results = []

best_score = -1
best_k = None
best_metric = None

print("\nValidation results:")
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

        print(f"metric={metric:10s} k={k:2d} val_acc={score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_metric = metric

print("\nBest model:")
print(f"metric = {best_metric}")
print(f"k      = {best_k}")
print(f"val_acc= {best_score:.4f}")

# ----------------------------
# 5. Plot results
# ----------------------------
plt.figure(figsize=(8, 5))

for metric in metrics:
    metric_results = [r for r in results if r["metric"] == metric]

    ks = [r["k"] for r in metric_results]
    accs = [r["accuracy"] for r in metric_results]

    plt.plot(ks, accs, marker='o', label=metric)

plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("kNN Hyperparameter Search (GTZAN)")
plt.xticks(list(k_values))
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# 6. Train final model
# ----------------------------
X_final = np.vstack([X_train, X_val])
y_final = np.concatenate([y_train, y_val])

final_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric=best_metric
)

final_knn.fit(X_final, y_final)

# ----------------------------
# 7. Final test evaluation
# ----------------------------
test_accuracy = final_knn.score(X_test, y_test)

print("\nFinal test performance:")
print(f"test_acc = {test_accuracy:.4f}")