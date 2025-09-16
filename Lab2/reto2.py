import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

iris = pd.read_csv(r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Iris.csv")
iris = iris.drop(columns=["Id"])

X = iris.drop("Species", axis=1)
y = LabelEncoder().fit_transform(iris["Species"])
labels = iris["Species"].unique()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=50)

models = {
    "Naive Bayes": GaussianNB(var_smoothing=1e-8),
    "SVM (RBF Kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma=0.1, random_state=50))
    ]),
    "SVM (Linear Kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", C=10, random_state=42))
    ])
}

fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
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

plt.figure(figsize=(8,5))
means = [results[m][0] for m in models.keys()]
stds = [results[m][1] for m in models.keys()]
plt.bar(models.keys(), means, yerr=stds, capsize=5, color=["#4c72b0", "#55a868", "#c44e52"])
plt.ylabel("Accuracy promedio (CV 5-fold)")
plt.title("Comparación de modelos")
plt.show()
