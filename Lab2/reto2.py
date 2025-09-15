import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# 1. Cargar dataset
# ===============================
iris = pd.read_csv(r"C:\Users\dmpie\Documentos\Python\VIsionIA\Lab2\Iris.csv")

iris = iris.drop(columns=["Id"])

# Separar features y labels
X = iris.drop("Species", axis=1)
y = LabelEncoder().fit_transform(iris["Species"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, stratify=y, random_state=42
)

models = {
    "Naive Bayes": GaussianNB(),
    "SVM (RBF Kernel)": Pipeline([
        ("scaler", StandardScaler()),  
        ("svm", SVC(kernel="rbf", C=1, gamma="scale", random_state=42))
    ]),
    "SVM (Linear Kernel)": Pipeline([
        ("scaler", StandardScaler()),   
        ("svm", SVC(kernel="linear", C=1, random_state=42))
    ])
}

for name, model in models.items():
    print(f"\n📌 {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas básicas
    print("Matriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=iris["Species"].unique()))


print("\n🔎 Validación cruzada (5-fold)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    print(f"{name}: Accuracy promedio={scores.mean():.3f}, desviación={scores.std():.3f}")
