#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : Percettrone                                                   |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 


"""
Questo tutorial è pensato per mostrare il funzionamento di un semplice percettrone  
"""
print(__doc__)


# importo librerie standard
import numpy as np
import matplotlib.pyplot as plt

# da sklearn importo solo dataset, usato per generare dei dati random e dividere i dati in train e test
from sklearn import datasets
from sklearn.model_selection import train_test_split


# importo la mia classe
from percettrone import *

# entry point
if __name__ == "__main__": 

    # Creo un dataset separabile linearmente con 2 features usando sklearn  
    dati, annotazioni = datasets.make_blobs(n_samples = 200, n_features=2, centers=2, random_state=50)  
    #np.random.seed(0) # imposto il random seed per essere sicuro di stare sempre nella stessa condizione
    #dati, annotazioni = datasets.make_blobs(n_samples = 100, centers = 2, cluster_std=1.5)
    
    # Divido il dataset in train e test 
    dati_train, dati_test, annotazioni_train, annotazioni_test = train_test_split(dati, annotazioni, test_size =0.3, random_state=1)

    # Faccio semplicemente la trasposta dei dati  
    dati_train = dati_train.T 
    dati_test = dati_test.T 
    num_input= dati_train.shape[0]
    print(num_input)

    plt.scatter(dati_train[0,:],dati_train[1,:], c="blue", alpha=0.6, label="train")# c=annotazioni_train )
    plt.scatter(dati_test[0,:],dati_test[1,:], c="red", alpha=0.6, label="test")# c=annotazioni_train )
    plt.title(" Dati generati per training e test ")
    plt.legend(loc="best")
    plt.xlabel("X1")
    plt.ylabel("Y2")
    plt.show()

    # istanziamo la classe percettrone
    percettrone_semplice = percettroneSemplice(_num_input=num_input, _num_iterazioni = 10000, _learning_rate =0.001)

    # ottimizziamo i parametri in base ai nostri dati di test 
    percettrone_semplice.ottimizza(dati_train, annotazioni_train)
    param_dict = percettrone_semplice.param_dict
    grad_dict = percettrone_semplice.grad_dict
    loss = percettrone_semplice.costi
    pesi = percettrone_semplice.pesi
    bias = percettrone_semplice.bias

    # Plotto la linea di separazione tra le classi
    valori_x1 = np.linspace(np.min(dati[:,0]),np.max(dati[:,0]),10).reshape((10,1))
    # i valori in y corrispondono alla retta data dai valori in x nella quale il bias è la quota e i pesi sono il coefficiente angolare 
    valori_x2 = -(pesi[0]*valori_x1 + bias)/(pesi[1]) 

    annotazioni_pred_train = percettrone_semplice.predict(dati_train)
    annotazioni_pred_test = percettrone_semplice.predict(dati_test)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(annotazioni_pred_train - annotazioni_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(annotazioni_pred_test - annotazioni_test)) * 100))

    plt.scatter(dati_train[0,:], dati_train[1,:], c=annotazioni_train, cmap="Dark2", alpha=0.6, label = "train" )
    plt.scatter(dati_test[0,:], dati_test[1,:], c=annotazioni_test, cmap="Dark2", alpha=1.0, marker = "^", label = "test")
    plt.plot(valori_x1, valori_x2, c="black", label = "Modello")
    plt.legend(loc="best")
    plt.title("Valutazione training e test")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    plt.plot(loss, c="blue", alpha=0.6)
    plt.title("Funzione di loss")
    plt.xlabel("Epoche x 1000")
    plt.ylabel("Valore della funzione di loss")
    plt.show()
  
