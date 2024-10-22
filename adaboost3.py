import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import unittest


class TestAdaboost(unittest.TestCase):

    def test_adaboost_predict(self):
        # Dane testowe
        X_train_sample = np.array([[1, 2], [1.5, 1.8], [2, 3], [2, 2.5]])
        y_train_sample = np.array([1, -1, 1, -1])

        # Trenowanie prawdziwego Decision Stump na danych
        stump1 = DecisionTreeClassifier(max_depth=1)
        stump1.fit(X_train_sample, y_train_sample)

        stump2 = DecisionTreeClassifier(max_depth=1)
        stump2.fit(X_train_sample, y_train_sample)

        classifiers = [stump1, stump2]
        alphas = [0.5, 0.8]

        # Dane testowe
        X_test_sample = np.array([[1, 2], [1.5, 1.8], [2, 3]])

        # Rzeczywisty wynik
        result = adaboost_predict(X_test_sample, classifiers, alphas)

        # Zgadywane wyniki przez Decision Stump na podstawie prostego problemu
        # Oczekiwane wyniki z wagami
        expected_result = np.sign(0.5 * stump1.predict(X_test_sample) + 0.8 * stump2.predict(X_test_sample))

        # Sprawdzenie, czy wynik jest zgodny z oczekiwaniami
        np.testing.assert_allclose(result, expected_result, atol=1e-6)


def adaboost_train(X_train, y_train, X_test, y_test, T):
    N, M = X_train.shape
    w = np.ones(N) / N
    classifiers = []
    alphas = []

    # Zmienna do śledzenia sumy predykcji
    final_predictions_train = np.zeros(N)

    for t in range(T):
        # Tworzenie i trenowanie Decision Stump (jednowęzłowe drzewo)
        stump = DecisionTreeClassifier(max_depth=1)
        stump.fit(X_train, y_train, sample_weight=w)
        y_pred_train = stump.predict(X_train)

        # Obliczanie błędu klasyfikatora
        error = np.sum(w * (y_pred_train != y_train)) / np.sum(w)
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

        # Aktualizacja wag dla przykładów
        w *= np.exp(-alpha * y_train * y_pred_train)
        w /= np.sum(w)

        # Dodanie klasyfikatora i jego wagi do listy
        classifiers.append(stump)
        alphas.append(alpha)

        # Aktualizacja bieżących predykcji dla zbioru treningowego
        final_predictions_train += alpha * y_pred_train
        y_train_pred = np.sign(final_predictions_train)

        # Obliczanie dokładności na zbiorze treningowym
        train_accuracy = np.mean(y_train_pred == y_train) * 100

        # Przewidywanie i obliczanie dokładności na zbiorze testowym
        y_test_pred = adaboost_predict(X_test, classifiers, alphas)
        test_accuracy = np.mean(y_test_pred == y_test) * 100

        # Wypisywanie dokładności po każdej iteracji
        print(f"Iteracja {t + 1}: Train Accuracy = {train_accuracy:.2f}%, Test Accuracy = {test_accuracy:.2f}%")

    return classifiers, alphas


def adaboost_predict(X, classifiers, alphas):
    final_predictions = np.zeros(X.shape[0])

    for classifier, alpha in zip(classifiers, alphas):
        final_predictions += alpha * classifier.predict(X)

    return np.sign(final_predictions)


# Generowanie danych Gaussa
X, y = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=1000)
y = np.where(y == 0, -1, 1)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Liczba iteracji AdaBoost
T = 10

# Trenowanie AdaBoost z użyciem Decision Stump
classifiers, alphas = adaboost_train(X_train, y_train, X_test, y_test, T)


# Wizualizacja granic decyzyjnych
def plot_decision_boundary(X, y, classifiers, alphas):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = adaboost_predict(np.c_[xx.ravel(), yy.ravel()], classifiers, alphas)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.show()


plot_decision_boundary(X_test, y_test, classifiers, alphas)

if __name__ == '__main__':
    unittest.main()
