#knn standard
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv("winequality.csv")  


if 'id' in df.columns:
    df = df.drop('id', axis=1)


def quality_class(q):
    if q <= 4:
        return 0  # slab
    elif q <= 6:
        return 1  # mediu
    else:
        return 2  # bun

df['quality_label'] = df['quality'].apply(quality_class)

X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(f"Acuratețea pentru k = {k}: {accuracy_score(y_test, y_pred):.4f}")
print("\nMatrice de confuzie:")
print(confusion_matrix(y_test, y_pred))
print("\nRaport de clasificare:")
print(classification_report(y_test, y_pred, target_names=['Slab', 'Mediu', 'Bun']))

#-------------------------------------
#knn cu CV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("winequality.csv")

if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Creează 3 clase (slab, mediu, bun)
def quality_class(q):
    if q <= 4:
        return 0
    elif q <= 6:
        return 1
    else:
        return 2

df['quality_label'] = df['quality'].apply(quality_class)

X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cross-validation pentru k = 1...30
k_values = list(range(1, 51))
acc_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, scoring='accuracy', cv=5)
    acc_scores.append(np.mean(scores))

best_k = k_values[np.argmax(acc_scores)]
best_acc = max(acc_scores)

print(f"\nCea mai bună valoare pentru k: {best_k}")
print(f"Acuratețea maximă estimată (CV): {best_acc:.4f}")

plt.style.use("ggplot")
plt.figure(figsize=(8, 5))
plt.plot(k_values, acc_scores, marker='o', color='blue')
plt.xlabel("Numărul de vecini (k)")
plt.ylabel("Acuratețea (Cross-Validation)")
plt.title("Efectul lui k asupra acurateței (CV)")
plt.xticks(k_values)
plt.grid(True)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

print("\n=== Evaluare finală pe setul de test ===")
print(f"Acuratețea pe test: {accuracy_score(y_test, y_pred):.4f}")
print("\nMatrice de confuzie:")
print(confusion_matrix(y_test, y_pred))
print("\nRaport de clasificare:")
print(classification_report(y_test, y_pred, target_names=['Slab', 'Mediu', 'Bun']))
#-----------------------------
#knn cu grid search
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("winequality.csv")

if 'id' in df.columns:
    df = df.drop('id', axis=1)

def quality_class(q):
    if q <= 4:
        return 0  # slab
    elif q <= 6:
        return 1  # mediu
    else:
        return 2  # bun

df['quality_label'] = df['quality'].apply(quality_class)

X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

t
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

k_values = list(range(1, 51))
param_grid = {'n_neighbors': k_values}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_k = grid_search.best_params_['n_neighbors']
best_cv_score = grid_search.best_score_

print(f"Cea mai bună valoare pentru k: {best_k}")
print(f"Acuratețea medie (CV) pentru k optim: {best_cv_score:.4f}")

best_model = KNeighborsClassifier(n_neighbors=best_k)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nAcuratețea pe setul de test: {test_accuracy:.4f}")
print("\nMatrice de confuzie:")
print(confusion_matrix(y_test, y_pred))
print("\nRaport de clasificare:")
print(classification_report(y_test, y_pred, target_names=['Slab', 'Mediu', 'Bun']))
#------------
#knn cu numpy
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("winequality.csv")


if 'id' in df.columns:
    df = df.drop('id', axis=1)

def quality_class(q):
    if q <= 4:
        return 0  # slab
    elif q <= 6:
        return 1  # mediu
    else:
        return 2  # bun

df['quality_label'] = df['quality'].apply(quality_class)


X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = list(range(1, 51))
acc_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, scoring='accuracy', cv=5)
    acc_scores.append(np.mean(scores))

best_k_index = np.argmax(acc_scores)
best_k = k_values[best_k_index]

print(f"Cea mai bună valoare pentru k: {best_k}")
print(f"Acuratețea medie (CV) pentru k optim: {acc_scores[best_k_index]:.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print(f"\nAcuratețea pe setul de test: {accuracy_score(y_test, y_pred):.4f}")
print("\nMatrice de confuzie:")
print(confusion_matrix(y_test, y_pred))
print("\nRaport de clasificare:")
print(classification_report(y_test, y_pred, target_names=['Slab', 'Mediu', 'Bun']))
#------
#curba
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.model_selection import ValidationCurve


df = pd.read_csv("winequality.csv")


if 'id' in df.columns:
    df = df.drop('id', axis=1)


def quality_class(q):
    if q <= 4:
        return 0  # slab
    elif q <= 6:
        return 1  # mediu
    else:
        return 2  # bun

df['quality_label'] = df['quality'].apply(quality_class)


X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


k_values = list(range(1, 101))


viz = ValidationCurve(
    KNeighborsClassifier(),
    param_name="n_neighbors",
    param_range=k_values,
    cv=5,
    scoring="accuracy"
)


viz.fit(X_scaled, y)
viz.show(outpath="Validation_Curve.png", dpi=100)

#-----------
#ponderi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

k_values = list(range(1, 51))
acc_weighted = []

# Calcul acuratețe cu distanțe ponderate
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_scaled, y, scoring='accuracy', cv=5)
    acc_weighted.append(np.mean(scores))

# Determină k optim
best_index = np.argmax(acc_weighted)
best_k = k_values[best_index]
best_score = acc_weighted[best_index]

# Print rezultate
print(f"Cea mai bună valoare pentru k (weighted KNN): {best_k}")
print(f"Acuratețe la k optim: {best_score:.4f}")

# Grafic
plt.style.use("ggplot")
plt.figure(figsize=(10, 6))
plt.plot(k_values, acc_weighted, marker='o', color='blue', label='Weighted KNN Accuracy')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
plt.xlabel("Numărul de vecini (k)")
plt.ylabel("Acuratețea medie (CV)")
plt.title("Efectul valorii lui k asupra acurateței (KNN cu distanțe ponderate)")
plt.legend()
plt.grid(True)
plt.savefig('weighted_knn_accuracy.png', dpi=100)
plt.show()

