import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Zbiór danych MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target.astype(int)

# Aby problem był binarny
binary_filter = (y == 0) | (y == 1)
X, y = X[binary_filter], y[binary_filter]

# Zmień etykiety na -1 i 1
y = np.where(y == 0, -1, 1)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Liczba iteracji
T = 10

# Implementacja AdaBoost
def adaboost_train(X_train, y_train, X_test, y_test, T):
    N, M = X_train.shape
    w = np.ones(N) / N
    classifiers = []
    alphas = []
    final_predictions_train = np.zeros(N)
    train_accuracies = []
    test_accuracies = []

    for t in range(T):
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X_train, y_train, sample_weight=w)
        y_pred_train = stump.predict(X_train)

        error = np.sum(w * (y_pred_train != y_train)) / np.sum(w)
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

        w *= np.exp(-alpha * y_train * y_pred_train)
        w /= np.sum(w)

        classifiers.append(stump)
        alphas.append(alpha)

        final_predictions_train += alpha * y_pred_train
        y_train_pred = np.sign(final_predictions_train)

        train_accuracy = np.mean(y_train_pred == y_train) * 100
        train_accuracies.append(train_accuracy)

        y_test_pred = adaboost_predict(X_test, classifiers, alphas)
        test_accuracy = np.mean(y_test_pred == y_test) * 100
        test_accuracies.append(test_accuracy)

        print(f"Iteracja {t + 1}: Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%")

    return classifiers, alphas, train_accuracies, test_accuracies

def adaboost_predict(X, classifiers, alphas):
    final_predictions = np.zeros(X.shape[0])
    for classifier, alpha in zip(classifiers, alphas):
        final_predictions += alpha * classifier.predict(X)
    return np.sign(final_predictions)

# Trenowanie własnej implementacji AdaBoost
classifiers, alphas, train_accuracies_own, test_accuracies_own = adaboost_train(X_train, y_train, X_test, y_test, T)

# Implementacja AdaBoost z biblioteki scikit-learn
train_accuracies_sklearn = []
test_accuracies_sklearn = []

for t in range(1, T + 1):
    sklearn_adaboost = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=t,
        algorithm="SAMME"  # SAMME jest odpowiednikiem algorytmu AdaBoost dla klasyfikacji binarnej
    )
    sklearn_adaboost.fit(X_train, y_train)

    y_train_pred_sklearn = sklearn_adaboost.predict(X_train)
    y_test_pred_sklearn = sklearn_adaboost.predict(X_test)

    train_accuracy_sklearn = accuracy_score(y_train, y_train_pred_sklearn) * 100
    test_accuracy_sklearn = accuracy_score(y_test, y_test_pred_sklearn) * 100

    train_accuracies_sklearn.append(train_accuracy_sklearn)
    test_accuracies_sklearn.append(test_accuracy_sklearn)

    print(f"Sklearn Iteracja {t}: Train Accuracy = {train_accuracy_sklearn:.2f}%, Test Accuracy = {test_accuracy_sklearn:.2f}%")

# Wykres porównujący dokładność obu wersji na zbiorze testowym
plt.figure(figsize=(12, 6))
plt.plot(range(1, T + 1), test_accuracies_own, label="Our AdaBoost Test Accuracy", marker='o')
plt.plot(range(1, T + 1), test_accuracies_sklearn, label="Sklearn AdaBoost Test Accuracy", marker='o', linestyle='--')
plt.xlabel("Number of Iterations")
plt.ylabel("Test Accuracy (%)")
plt.title("Comparison of Test Accuracy between Our AdaBoost and Sklearn's AdaBoost")
plt.legend()
plt.grid(True)
plt.show()
