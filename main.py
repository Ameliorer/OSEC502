# pour les datasets et randomForest
import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
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


def main():
    # =============================
    #          DATA SET
    # =============================

    # division du dataset en deux partie, test et entrainement
    X, y = recup('data')

    X_train = X
    y_train = y


    # 30% test et 70% entrainement
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    #X_test, y_test = hsv_hist('photo447.png', 'eau')

    # affichage des éléments obtenus
    print(f'X : ${X}')
    print(f'Y : ${y}')
    print(f'X_train : ${X_train}')
    print(f'X_test : ${X_test}')
    print(f'y_train : ${y_train}')
    print(f'y_test : ${y_test}')
    print(type(X_test))
    print(type(y_test))

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
    print("\nSCORE OF THE MODEL: ", clf.score(X_test, y_test))
    print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred))

    # affichage d'une matrice pour mieux comprendre la précision
    plt.figure(figsize=(10, 7))
    sn.heatmap(confusion_matrix(y_test, y_pred), annot=True)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.show()

    # Prédiction sur l'ensemble de test
    predictions = clf.predict(X_test)
    print("predictions", predictions)

    plt.show()

    # Prédire quel type de fleur c'est.
    # print(clf.predict([[3, 3, 2, 2]]))


def image():
    # =============================
    #            IMAGE
    # =============================

    # ouverture de l'image photo.png
    img = cv.imread('photo447.png')
    # récupération du nombre de colonnes et de lignes
    rows, cols, tests = img.shape

    # parcours de tous les pixels
    for i in range(rows):
        for j in range(cols):
            pixel = image[i][j]
            pixel_b, pixel_g, pixel_r = pixel
            if (pixel_b > pixel_g) and (pixel_b > pixel_r):
                image[i][j] = [255, 0, 0]
            else:
                image[i][j] = [0, 0, 0]
            print(image[i][j])

    # L'histogramme
    chans = cv.split(img)
    colors = ("b", "g", "r")
    plt.figure()
    plt.title("'Flattened' Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    # loop over the image channels
    for (chan, color) in zip(chans, colors):
        # create a histogram for the current channel and plot it
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
        plt.show()

    # nom des fenêtres
    window_name = 'image'
    window_name1 = 'image grey'

    # Changement des couleurs de l'image en nuances de gris.
    # y = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # affichage des fenêtres avec un nom et une image
    # cv.imshow(window_name, img)
    # cv.imshow(window_name1, y)

    # attends l'appuie d'une touche (pour éviter que python crash)
    cv.waitKey(0)

    # ferme toutes les fenêtres
    cv.destroyAllWindows()
# PAS UTILISE

def main2():
    # ouverture de l'image photo.png
    img = cv.imread('photo447.png')
    # récupération du nombre de colonnes et de lignes
    rows, cols, tests = img.shape

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Set the lower and upper bounds for the blue hue
    lower_blue = np.array([90, 100, 50])
    upper_blue = np.array([130, 255, 255])

    # create a mask for green colour using inRange function
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # perform bitwise and on the original image arrays using the mask
    res = cv.bitwise_and(img, img, mask=mask)

    window_name = 'image'
    cv.imshow(window_name, img)

    window_name = 'image saturee'
    cv.imshow(window_name, hsv)

    window_name = 'image bleu'
    cv.imshow(window_name, res)

    # attends l'appuie d'une touche (pour éviter que python crash)
    cv.waitKey(0)

    # ferme toutes les fenêtres
    cv.destroyAllWindows()
# PAS UTILISE

def hsv_hist(nom_fichier, label):
    img = cv.imread(nom_fichier)
    img2 = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]

    hist_h = cv.calcHist([h], [0], None, [10], [0, 256])
    hist_s = cv.calcHist([s], [0], None, [10], [0, 256])
    hist_v = cv.calcHist([v], [0], None, [10], [0, 256])
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.show()

    values = []
    mat = np.concatenate((hist_h, hist_s, hist_v), axis=1)  # on concatène les histogrammes de l'image.
    mat = mat.astype(int)
    values = np.array(mat.flatten())  # on ajoute les valeurs trouvées dans le fichier au tableau final.

    return [values], [label]


if __name__ == "__main__":
    # print (hsv_hist('photo447.png'))
    main()
    # recup()

# faire un histogramme hsv avec 10 valeurs sur un lac, une foret, etc.
# donner ces histogrammes au random forest
