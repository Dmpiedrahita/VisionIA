# iris_kmeans_mlp_experiment.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, classification_report,
                             adjusted_rand_score, silhouette_score,
                             homogeneity_score, completeness_score)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from joblib import dump

# -----------------------
OUT_DIR = "results_iris_kmeans_mlp"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# 0. Cargar datos
# -----------------------
iris = pd.read_csv(r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Iris.csv")
# quitar Id si existe
if "Id" in iris.columns:
    iris = iris.drop(columns=["Id"])

X = iris.drop(columns=["Species"]).values
y = LabelEncoder().fit_transform(iris["Species"])
labels_true = LabelEncoder().fit_transform(iris["Species"])
class_names = LabelEncoder().fit(iris["Species"]).classes_

# Escalar (importante para KMeans y MLP)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Guardar scaler
dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

# -----------------------
# 1. Método del codo + silhouette para K (inertia + silhouette)
# -----------------------
ks = list(range(1,11))
inertias = []
silhouettes = []

for k in ks:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    else:
        silhouettes.append(np.nan)

# Plot: inertia (codo) y silhouette
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(ks, inertias, '-o')
plt.title("Método del codo (Inertia)")
plt.xlabel("k")
plt.xticks(ks)
plt.ylabel("Inertia (Suma de distancias al cuadrado)")

plt.subplot(1,2,2)
plt.plot(ks, silhouettes, '-o')
plt.title("Silhouette score por k")
plt.xlabel("k")
plt.xticks(ks)
plt.ylabel("Silhouette score")
plt.ylim(-0.1,1.0)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "elbow_and_silhouette.png"), dpi=150)
plt.show()

# -----------------------
# 2. Selección K y evaluación detallada
# -----------------------
# Según iris clásico, k=3 suele ser lo esperado; dejamos la selección flexible:
k_sel = 3
kmeans = KMeans(n_clusters=k_sel, random_state=42, n_init=50)
kmeans.fit(X_scaled)
clusters = kmeans.labels_

# Métricas de clustering (no supervisadas)
ari = adjusted_rand_score(labels_true, clusters)
sil = silhouette_score(X_scaled, clusters)
hom = homogeneity_score(labels_true, clusters)
comp = completeness_score(labels_true, clusters)

metrics_clust = {
    "k": k_sel,
    "adjusted_rand_index": ari,
    "silhouette": sil,
    "homogeneity": hom,
    "completeness": comp
}
pd.DataFrame([metrics_clust]).to_csv(os.path.join(OUT_DIR, f"kmeans_metrics_k{k_sel}.csv"), index=False)

print("KMeans metrics (k={}):".format(k_sel))
print(metrics_clust)

# Matriz de confusión entre clusters y etiquetas (map clusters -> labels por mayoría)
def cluster_to_label_map(y_true, clusters):
    mapping = {}
    for c in np.unique(clusters):
        mask = clusters == c
        true_labels_in_c = y_true[mask]
        if len(true_labels_in_c)==0:
            mapping[c] = -1
        else:
            # asignar la clase más frecuente en ese cluster
            mapping[c] = np.bincount(true_labels_in_c).argmax()
    return mapping

mapping = cluster_to_label_map(labels_true, clusters)
pred_labels_from_clusters = np.array([mapping[c] for c in clusters])

cm = confusion_matrix(labels_true, pred_labels_from_clusters)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusión (clusters k={k_sel} -> clases)")
plt.xlabel("Predicho por cluster->clase")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"cm_kmeans_k{k_sel}.png"), dpi=150)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, stratify=y, random_state=42)
w
mlp = MLPClassifier(random_state=42, max_iter=500)  

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

print("\nMLP (base) classification report:")
print(classification_report(y_test, y_pred, target_names=class_names))
cm_mlp = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm_mlp, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("MLP - matriz de confusión (base)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cm_mlp_base.png"), dpi=150)
plt.show()

param_grid_mlp = {
    "hidden_layer_sizes": [(10,), (20,), (10,10), (20,10)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],  # regularización L2
    "learning_rate_init": [0.001, 0.01],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_mlp = GridSearchCV(MLPClassifier(random_state=42, max_iter=2000), param_grid_mlp, cv=skf, scoring="accuracy", n_jobs=-1, verbose=0)
grid_mlp.fit(X_train, y_train) 

print("Mejores params MLP:", grid_mlp.best_params_)
print("Mejor score CV (MLP):", grid_mlp.best_score_)

pd.DataFrame(grid_mlp.cv_results_).to_csv(os.path.join(OUT_DIR, "grid_mlp_results.csv"), index=False)
best_mlp = grid_mlp.best_estimator_
dump(best_mlp, os.path.join(OUT_DIR, "best_mlp.joblib"))

y_pred_best = best_mlp.predict(X_test)
print("\nMLP (best) classification report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm_best = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5,4))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("MLP - matriz de confusión (best)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cm_mlp_best.png"), dpi=150)
plt.show()


cv_scores_mlp = cross_val_score(best_mlp, X_scaled, y, cv=skf, scoring="accuracy", n_jobs=-1)
print(f"MLP CV mean={cv_scores_mlp.mean():.4f}, std={cv_scores_mlp.std():.4f}")
pd.DataFrame({"mlp_cv_scores": cv_scores_mlp}).to_csv(os.path.join(OUT_DIR, "mlp_cv_scores.csv"), index=False)

ari_list = []
for seed in [42, 0, 7, 21, 100]:
    km = KMeans(n_clusters=k_sel, random_state=seed, n_init=50).fit(X_scaled)
    ari_list.append(adjusted_rand_score(labels_true, km.labels_))
pd.DataFrame({"ari": ari_list}).to_csv(os.path.join(OUT_DIR, "kmeans_ari_seeds.csv"), index=False)
print("KMeans ARI (varias semillas):", ari_list)

def save_classif_report(report_dict, fname):
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(os.path.join(OUT_DIR, fname), index=True)

save_classif_report(classification_report(y_test, y_pred, target_names=class_names, output_dict=True), "classif_report_mlp_base.csv")
save_classif_report(classification_report(y_test, y_pred_best, target_names=class_names, output_dict=True), "classif_report_mlp_best.csv")

pd.DataFrame([metrics_clust]).to_csv(os.path.join(OUT_DIR, "kmeans_metrics_summary.csv"), index=False)

print("\nTodos los resultados se han guardado en:", OUT_DIR)
