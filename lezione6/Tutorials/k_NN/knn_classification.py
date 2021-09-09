#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : k-Nearest Neighbors                                           |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Esempio di classificazione che usa il k-nearest neighbors
Questo esempio semplicemente plotterà i confini per ogni classe indicata negli esempi generati in maniera random

"""
print(__doc__)

# importiamo le librerie necessarie, 
# numpy per operazioni matematiche, 
# matplotlib per la grafica
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# sklearn è una collezione di funzioni utili 
# per algoritmi di machine learning in python
from sklearn import neighbors, datasets
from sklearn.datasets.samples_generator import make_blobs 

# imposto il numero di neighbors
n_neighbors = 15
passo_mesh = .02  # passo per la mesh

# crea dati random usando un generatore di sklearn, i dati sono composti 
# n_samples = 40 campioni 
# aventi 2 features (quindi 2 dimensioni)
# e 2 possibili annotazioni (quindi 2 centroidi) centers = 2
# random_state ci serve a cambiare i campioni generati come un seed, proviamo valori 50 e 40 
dati, annotazioni = make_blobs(n_samples = 40, n_features=2, centers=4, random_state=50)  

# Creo le color maps
mesh_color_map = ListedColormap(['orange', 'cyan', 'red','cornflowerblue'])

# possiamo eseguire l'esercizio 2 volte, prima usando l'opzione "uniform" sui pesi e dopo usando "distance"
weights = 'uniform'#'distance'
# Istanziamo il classificatore NN e fittiamo i dati
clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(dati, annotazioni)

# plottiamo i confini decisionali con colori diversi per ogni punto della mesh
x_min, x_max = dati[:, 0].min() - 1, dati[:, 0].max() + 1
y_min, y_max = dati[:, 1].min() - 1, dati[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, passo_mesh),
                        np.arange(y_min, y_max, passo_mesh))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Coloriamo i risultati
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=mesh_color_map)

# Grafichiamo gli esempi di training
sns.scatterplot(x=dati[:, 0], y=dati[:, 1], 
                alpha=1.0, edgecolor="black")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Classificazione 3-Classi (k = %i, pesi = '%s')"
            % (n_neighbors, weights))
plt.xlabel("X")
plt.ylabel("Y")

plt.show()