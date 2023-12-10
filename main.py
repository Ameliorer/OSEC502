# pour les datasets et randomForest
import numpy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
# pour les images
import cv2 as cv
import numpy as np

# pour les fichiers
import os


def recup(nomDossier):
    """
    Fonction qui récupère les données dans les fichiers qui sont dans le dossier nomDossier.
    Les données sont les valeurs des histogrammes HSV des images.
    :param nomDossier: String, le nom du dossier.
    :return: values, labels : les valeurs et les labels qui se trouvent dans les fichiers.
    """
    values = []
    labels = []
    # on ouvre le dossier et on récupère les noms des fichiers.
    for filename in os.listdir(nomDossier):
        # pour chaque fichier, on l'ouvre.
        with open(os.path.join(nomDossier, filename)) as f:
            mat = []  # la matrice qui contiendra les valeurs de l'histogramme HSV.
            content = f.readline()  # on lit chaque ligne du fichier (ici premiere ligne).
            labels.append(content.split("\n")[0])  # on récupère le nom du label (premiere ligne).
            content = f.readline()  # on lit la deuxième ligne du fichier.

            # tant qu'on a des lignes à lire
            while content:
                line = content.split("\n")[0].split(",")  # on sépare les données pour obtenir seulement les chiffres
                # (pas d'espaces ni de retour à la ligne).
                line = [float(i) for i in line]  # on crée un tableau qui contient tous les chiffres de la ligne.
                mat.append(line)  # on ajoute cette ligne dans la matrice.
                content = f.readline()  # on lit la ligne suivante.

        mat = numpy.concatenate(mat)  # on concatène tous les lignes de la matrice pour en obtenir qu'une seule.
        f.close()  # on ferme le fichier.
        values.append(mat)  # on ajoute les valeurs trouvées dans le fichier au tableau final.

    # print(numpy.array(labels))
    # print(numpy.array(values))


    return values, labels

from sklearn.tree import plot_tree

def main():
    def plot_trees(classifier, num_trees=3):
        # Accéder aux trois premiers arbres
        three_trees = classifier.estimators_[:num_trees]

        # Afficher graphiquement chaque arbre dans une figure distincte
        for index, tree in enumerate(three_trees):
            plt.figure(figsize=(10, 7))
            plot_tree(tree, filled=True, class_names=y_train, feature_names=[f'feature {i}' for i in range(
                len(X_train[0]))])  # Remplacez feature_names par vos noms de fonctionnalités si disponibles
            plt.title(f"Arbre {index + 1}")
            plt.show()
    # =============================
    #          DATA SET
    # =============================
    # division du dataset en deux partie, test et entrainement
    X, y = recup('data')

    # 30% test et 70% entrainement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # =============================
    #   RANDOM FOREST CLASSIFIER
    # =============================
    # creation du classifier Random Forest
    # n_estimators est le nombre d'arbres d'entrainement
    clf = RandomForestClassifier(n_estimators=100)

    # entraine le model sur le dataset
    # "fit" utilise les paramètres comme set d'entrainement
    clf.fit(X_train, y_train)

    # "y_pred" est la prédiction du model sur le dataset de test
    y_pred = clf.predict(X_test)

    # affichage du score du model et de la précision du model
    print("\nSCORE DU MODEL: ", clf.score(X_test, y_test))

    # affichage d'une matrice pour mieux comprendre la précision
    classe = ["Foret", "Lac", "Desert"]
    plt.figure(figsize=(10, 7))
    sn.heatmap(confusion_matrix(y_test, y_pred), annot=True, xticklabels=classe, yticklabels=classe)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.show()

    plot_trees(clf, num_trees=3)



def hsv_hist(nom_fichier, label):
    """
     Fonction qui récupère l’histogramme HSV de l’image.
    :param nom_fichier: String, le nom du fichier.
    :param label: String, ce que l’image représente .
     :return: values, labels : les valeurs et les labels qui se trouvent dans l’image.
     """
    # Lecture de l’image
    img = cv.imread(nom_fichier)
    img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Récupération des bandes
    h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]

    # Calcul des histogrammes et affichage de la matrice
    hist_h = cv.calcHist([h], [0], None, [10], [0, 256])
    hist_s = cv.calcHist([s], [0], None, [10], [0, 256])
    hist_v = cv.calcHist([v], [0], None, [10], [0, 256])
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.show()

    # Concaténation
    values = []
    mat = np.concatenate((hist_h, hist_s, hist_v), axis=1)  # on concatène les histogrammes de l'image.
    mat = mat.astype(int)
    values = np.array(mat.flatten())  # on ajoute les valeurs trouvées dans le fichier au tableau final.


    return [values], [label]

if __name__ == "__main__":
    main()