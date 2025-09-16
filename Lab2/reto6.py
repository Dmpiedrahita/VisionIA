import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.io import imread
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score

datadir = r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2"
categories = ["coast", "forest", "highway"]

features, labels = [], []
print("🔎 Cargando imágenes y extrayendo características...")
for idx, cat in enumerate(categories):
    path = os.path.join(datadir, cat)
    for img_file in os.listdir(path):
        img = imread(os.path.join(path, img_file))
        if img.ndim == 3:
            r_mean, g_mean, b_mean = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
            r_std, g_std, b_std = img[:, :, 0].std(), img[:, :, 1].std(), img[:, :, 2].std()
            features.append([r_mean, g_mean, b_mean, r_std, g_std, b_std])
            labels.append(idx)

X = np.array(features)
y = np.array(labels)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM (Linear Kernel)": SVC(kernel="linear", random_state=42),
    "SVM (RBF Kernel)": SVC(kernel="rbf", random_state=42),
    "MLP (base)": MLPClassifier(random_state=42, max_iter=1000),
    "MLP (tuned)": MLPClassifier(
        random_state=42,
        max_iter=1500,
        hidden_layer_sizes=(20, 10),
        activation="tanh",
        alpha=0.01,
        learning_rate_init=0.001
    )
}

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=categories, yticklabels=categories, ax=axes[idx])

    acc = (y_pred == y_test).mean()
    axes[idx].set_title(f"{name}\nAcc={acc:.2f}", fontsize=9)
    axes[idx].set_xlabel("Predicho", fontsize=7)
    axes[idx].set_ylabel("Real", fontsize=7)
    axes[idx].tick_params(axis='x', labelsize=7)
    axes[idx].tick_params(axis='y', labelsize=7)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
y_kmeans = kmeans.fit_predict(X_test)

labels_map = {}
for i in range(3):
    mask = (y_kmeans == i)
    if np.any(mask):
        labels_map[i] = mode(y_test[mask], keepdims=True)[0][0]

y_pred_kmeans = np.array([labels_map[c] for c in y_kmeans])
cm = confusion_matrix(y_test, y_pred_kmeans)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=categories, yticklabels=categories, ax=axes[len(models)])

acc = (y_pred_kmeans == y_test).mean()
axes[len(models)].set_title(f"KMeans\nAcc={acc:.2f}", fontsize=9)
axes[len(models)].set_xlabel("Predicho", fontsize=7)
axes[len(models)].set_ylabel("Real", fontsize=7)
axes[len(models)].tick_params(axis='x', labelsize=7)
axes[len(models)].tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.show()


print("\n🔎 Validación cruzada (5-fold)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
    results[name] = (scores.mean(), scores.std())
    print(f"{name}: Accuracy promedio={scores.mean():.3f}, desviación={scores.std():.3f}")

kmeans_scores = []
for train_idx, test_idx in skf.split(X_scaled, y):
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    km = KMeans(n_clusters=3, random_state=42, n_init=20)
    y_pred_km = km.fit_predict(X_te)

    labels_map = {}
    for i in range(3):
        mask = (y_pred_km == i)
        if np.any(mask):
            labels_map[i] = mode(y_te[mask], keepdims=True)[0][0]

    y_pred_km = np.array([labels_map[c] for c in y_pred_km])
    kmeans_scores.append(accuracy_score(y_te, y_pred_km))

results["KMeans (k=3)"] = (np.mean(kmeans_scores), np.std(kmeans_scores))
print(f"KMeans (k=3): Accuracy promedio={np.mean(kmeans_scores):.3f}, desviación={np.std(kmeans_scores):.3f}")

plt.figure(figsize=(10, 5))
means = [results[m][0] for m in results.keys()]
stds = [results[m][1] for m in results.keys()]
plt.bar(results.keys(), means, yerr=stds, capsize=5, color=sns.color_palette("Set2", len(results)))
plt.ylabel("Accuracy promedio (CV 5-fold)")
plt.xticks(rotation=45, ha="right")
plt.title("Comparación de modelos en clasificación de escenas")
plt.tight_layout()
plt.show()
