import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump

OUT_DIR = "results_knn_gb"
os.makedirs(OUT_DIR, exist_ok=True)

iris = pd.read_csv(r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Iris.csv")

if "Id" in iris.columns:
    iris = iris.drop(columns=["Id"])

X = iris.drop(columns=["Species"]).values
y = LabelEncoder().fit_transform(iris["Species"])
class_names = LabelEncoder().fit(iris["Species"]).classes_

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33,
                                                    stratify=y, random_state=42)

models = {
    "k-NN": KNeighborsClassifier(),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📌 {name} - Reporte clasificación")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{name} - Matriz de Confusión")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"cm_{name}.png"), dpi=150)
    plt.show()

param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
param_grid_gb = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [2, 3, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=cv, scoring="accuracy", n_jobs=-1)
grid_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=cv, scoring="accuracy", n_jobs=-1)

grid_knn.fit(X_train, y_train)
grid_gb.fit(X_train, y_train)

print("\n🔎 Mejor KNN:", grid_knn.best_params_, "Accuracy CV:", grid_knn.best_score_)
print("🔎 Mejor GB:", grid_gb.best_params_, "Accuracy CV:", grid_gb.best_score_)

best_knn = grid_knn.best_estimator_
best_gb = grid_gb.best_estimator_


for name, model in [("Best k-NN", best_knn), ("Best GB", best_gb)]:
    y_pred = model.predict(X_test)
    print(f"\n⭐ {name} - Reporte clasificación (optimizado)")
    print(classification_report(y_test, y_pred, target_names=class_names))

dump(best_knn, os.path.join(OUT_DIR, "best_knn.joblib"))
dump(best_gb, os.path.join(OUT_DIR, "best_gb.joblib"))
pd.DataFrame(grid_knn.cv_results_).to_csv(os.path.join(OUT_DIR, "grid_knn_results.csv"), index=False)
pd.DataFrame(grid_gb.cv_results_).to_csv(os.path.join(OUT_DIR, "grid_gb_results.csv"), index=False)

print("\n✅ Resultados guardados en:", OUT_DIR)
