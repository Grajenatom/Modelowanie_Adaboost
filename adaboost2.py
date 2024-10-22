import numpy as np
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import unittest
import numpy as np


class TestAdaboost(unittest.TestCase):

    def test_adaboost_predict(self):
        # Dane testowe
        X_test_sample = np.array([[1, 2], [1.5, 1.8], [2, 3]])
        y_test_sample = np.array([1, -1, 1])

        # Sztuczne klasyfikatory KNN i ich wagi alphas (dla uproszczenia przykładu)
        class FakeClassifier:
            def predict(self, X):
                # Zwraca stałe wartości (dla prostoty zawsze 1)
                return np.array([1 for _ in X])

        classifiers = [FakeClassifier(), FakeClassifier()]
        alphas = [0.5, 0.8]

        # Oczekiwane wyniki
        expected_result = np.sign(0.5 * np.array([1, 1, 1]) + 0.8 * np.array([1, 1, 1]))

        # Rzeczywisty wynik
        result = adaboost_predict(X_test_sample, classifiers, alphas)

        # Sprawdzenie, czy wynik jest zgodny z oczekiwaniami z tolerancją na dokładność numeryczną
        np.testing.assert_allclose(result, expected_result, atol=1e-6)


def adaboost_train(X_train, y_train, X_test, y_test, T, n_neighbors=3):
    N, M = X_train.shape
    w = np.ones(N) / N
    classifiers = []
    alphas = []
    
    # Zmienna do śledzenia sumy predykcji
    final_predictions_train = np.zeros(N)
    
    for t in range(T):
        # Tworzenie i trenowanie klasyfikatora KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        y_pred_train = knn.predict(X_train)
        
        # Obliczanie błędu klasyfikatora
        error = np.sum(w * (y_pred_train != y_train)) / np.sum(w)
        alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
        
        # Aktualizacja wag dla przykładów
        w *= np.exp(-alpha * y_train * y_pred_train)
        w /= np.sum(w)
        
        # Dodanie klasyfikatora i jego wagi do listy
        classifiers.append(knn)
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
n_neighbors = 3  # Liczba sąsiadów w KNN

# Trenowanie AdaBoost z użyciem KNN
classifiers, alphas = adaboost_train(X_train, y_train, X_test, y_test, T, n_neighbors)

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