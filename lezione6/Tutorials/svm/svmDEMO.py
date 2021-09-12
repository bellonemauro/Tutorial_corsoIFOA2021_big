#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : Support vector machines                                       |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 


"""
Questo tutorial è pensato per mostrare il funzionamento di un algoritmo di classificazione
basato su support vector machine SVM

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm 
from sklearn.datasets.samples_generator import make_blobs 

# crea dati random usando un generatore di sklearn, i dati sono composti 
# n_samples = 40 campioni 
# aventi 2 features (quindi 2 dimensioni)
# e 2 possibili annotazioni (quindi 2 centroidi) centers = 2
# random_state ci serve a cambiare i campioni generati come un seed, proviamo valori 50 e 40 
dati, annotazioni = make_blobs(n_samples = 40, n_features=2, centers=2, random_state=50)  

# stampiamo i dati generati 40x2
print(dati)
input("Dati generati, premi invio per continuare \n\n")

# stampiamo il vettore delle annotazioni 40x1, per ogni dato c'è una annotazione 
print(annotazioni)
input("Annotazioni generati, premi invio per continuare \n\n")

# istanziamo il modello del classificatore 
# possiamo cambiare il kernel usando delle altre funzioni come rbf (radial based function)
classificatore = svm.SVC(kernel='linear', C=1)
print (classificatore)
input ("Attributi del classificatore istanziato, premi invio per continuare \n\n")

# la funzione fit esegue l'algoritmo di fittaggio dei dati quindi generazione del classificatore ottimo 
classificatore.fit(dati, annotazioni)
print (classificatore.support_vectors_)
input ("Vettori di supporto generati, premi invio per continuare \n\n ")


# visualiziamo i dati inserendo come colore l'annotazione (ground truth) generata
plt.scatter(dati[:,0], dati[:,1], c=annotazioni, s=30, cmap=plt.cm.Paired)
# oggetto grafico per manipolare gli assi
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

#crea una griglia per valutare il modello
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy,xx)


# plottiamo la funzione di decisione
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = classificatore.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5, linestyles=['--', '-','--'])
# plottiamo i vettori supporto
ax.scatter(classificatore.support_vectors_[:,0], classificatore.support_vectors_[:,1], s=100, linewidth=1, facecolors='none')
#plt.show()

# testiamo il classificatore su un nuovo dato precedentemente non visto
dati_test, annotazioni_vere = make_blobs(n_samples = 10, n_features=2, centers=2, random_state=50)
#dati_test, annotazioni_vere = make_blobs(n_samples = 1000, n_features=2, centers=1, random_state=49)
annotazioni_generate = classificatore.predict(dati_test)
plt.scatter(dati_test[:,0], dati_test[:,1], c=annotazioni_generate, s=30, cmap=plt.cm.Accent)
plt.show()

