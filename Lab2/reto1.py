import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

iris = pd.read_csv(r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Iris.csv")

X = iris.drop("Species", axis=1)
le = LabelEncoder()
y = le.fit_transform(iris["Species"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10, random_state=4))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=10, max_depth=4)
}

fig, axes = plt.subplots(1, len(models), figsize=(15, 4))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(name)
    ax.set_ylabel("Etiqueta real")
    ax.set_xlabel("Predicción")

    print(f"\n📌 {name}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

plt.tight_layout()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    results[name] = scores.mean()
    print(f"{name}: Accuracy promedio={scores.mean():.3f}, desviación={scores.std():.3f}")

plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Accuracy promedio por modelo (CV=5)")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()