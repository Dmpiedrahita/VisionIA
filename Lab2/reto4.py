import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

iris = pd.read_csv(r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Iris.csv")
if "Id" in iris.columns:
    iris = iris.drop(columns=["Id"])

X = iris.drop("Species", axis=1).values
y = LabelEncoder().fit_transform(iris["Species"])
labels = LabelEncoder().fit(iris["Species"]).classes_

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, stratify=y, random_state=42
)

models = {
    "k-NN": KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",
    metric="minkowski",
    p=2
),
    "GradientBoosting": GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=1,
    random_state=42
)
}

fig, axes = plt.subplots(1, len(models), figsize=(12, 5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    print(f"\n📌 {name} - Reporte clasificación")
    print(classification_report(y_test, y_pred, target_names=labels))

plt.tight_layout()
plt.show()

print("\n🔎 Validación cruzada (5-fold)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
    results[name] = (scores.mean(), scores.std())
    print(f"{name}: Accuracy promedio={scores.mean():.3f}, desviación={scores.std():.3f}")

plt.figure(figsize=(8, 5))
means = [results[m][0] for m in models.keys()]
stds = [results[m][1] for m in models.keys()]
plt.bar(models.keys(), means, yerr=stds, capsize=5,
        color=["#4c72b0", "#55a868"])
plt.ylabel("Accuracy promedio (CV 5-fold)")
plt.title("Comparación de modelos")
plt.show()
