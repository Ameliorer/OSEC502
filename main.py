# pour les datasets et randomForest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix


def main():
    # =============================
    #          DATA SET
    # =============================

    # Chargement du dataset
    iris = datasets.load_iris()
    print(iris.target_names)
    print(iris.feature_names)

    # division du dataset en deux partie, test et entrainement
    X, y = datasets.load_iris(return_X_y=True)

    # 30% test et 70% entrainement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # creation du dataFrame avec Panda
    data = pd.DataFrame({'sepallength': iris.data[:, 0], 'sepalwidth': iris.data[:, 1],
                         'petallength': iris.data[:, 2], 'petalwidth': iris.data[:, 3],
                         'species': iris.target})

    # affichage du dataset (le top 5)
    print(data.head())

    # =============================
    #   RANDOM FOREST CLASSIFIER
    # =============================

    # creation du classifier Random Forest
    # n_estimators est le nombre d'arbre d'entrainement
    clf = RandomForestClassifier(n_estimators=100)

    # entraine le model sur le dataset
    # fit utilise les parametres comme set d'entrainement
    clf.fit(X_train, y_train)

    # y_pred est la prédiction du model sur le dataset de test
    y_pred = clf.predict(X_test)

    # affichage du score du model et de la précision du model
    print("\nSCORE OF THE MODEL: ", clf.score(X_test, y_test))
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    # affichage d'une matrice pour mieux comprendre la précision
    plt.figure(figsize=(10,7))
    sn.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.show()

    # predicting which type of flower it is.
    #print(clf.predict([[3, 3, 2, 2]]))


if __name__ == "__main__":
    main()