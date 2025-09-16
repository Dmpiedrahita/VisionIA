import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, adjusted_rand_score, silhouette_score

iris = pd.read_csv(r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Iris.csv")
iris = iris.drop(columns=["Id"])

X = iris.drop("Species", axis=1)
y = LabelEncoder().fit_transform(iris["Species"])
labels = iris["Species"].unique()

scaler = StandardScaler()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=42)

models = {
    "KMeans (k=3)": Pipeline([
        ("scaler", scaler),
        ("kmeans", KMeans(n_clusters=3, random_state=21, n_init=1000))
    ]),
    "MLP (base)": Pipeline([
        ("scaler", scaler),
        ("mlp", MLPClassifier(random_state=50, max_iter=2000))
    ]),
    "MLP (tuned)": Pipeline([
        ("scaler", scaler),
        ("mlp", MLPClassifier(
            random_state=42,
            max_iter=1500,
            hidden_layer_sizes=(20, 10),
            activation="tanh",
            alpha=0.01,
            learning_rate_init=0.001
        ))
    ])
}

fig, axes = plt.subplots(1, len(models), figsize=(15, 5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

plt.tight_layout()
plt.show()

print("\n🔎 Validación cruzada (5-fold)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    results[name] = (scores.mean(), scores.std())
    print(f"{name}: Accuracy promedio={scores.mean():.3f}, desviación={scores.std():.3f}")

plt.figure(figsize=(8, 5))
means = [results[m][0] for m in models.keys()]
stds = [results[m][1] for m in models.keys()]
plt.bar(models.keys(), means, yerr=stds, capsize=5,
        color=["#4c72b0", "#55a868", "#c44e52"])
plt.ylabel("Accuracy promedio (CV 5-fold)")
plt.title("Comparación de modelos")
plt.show()

inercia = []
K = range(1, 10) 

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=21, n_init=50)
    kmeans.fit(scaler.fit_transform(X)) 
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inercia, "o-", color="red")
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inercia (Suma de distancias cuadradas)")
plt.title("Método del codo")
plt.show()


