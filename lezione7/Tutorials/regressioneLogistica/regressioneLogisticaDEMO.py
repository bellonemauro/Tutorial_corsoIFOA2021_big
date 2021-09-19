#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : regessione polinomiale                                        |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Esempio di regressione logistica
"""
print(__doc__)

# importo librerie standard matematica e plotting
import numpy as np
import matplotlib.pyplot as plt

# importiamo gli strumenti che ci servono da sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# entry point
if __name__ == "__main__": 

    # creo dei vettori di dati un un target binario 0,1
    dati = np.arange(10).reshape(-1, 1)
    annotazioni_ground_truth = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    print(dati)
    print(annotazioni_ground_truth)
    input ("Dati e annotazioni generate, premi invio per continuare")

    # istanzio il modello di regressione logistica
    model = LogisticRegression(solver='liblinear', random_state=0)
    #LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #                   intercept_scaling=1, l1_ratio=None, max_iter=100,
    #                   multi_class='warn', n_jobs=None, penalty='l2',
    #                   random_state=0, solver='liblinear', tol=0.0001, verbose=0,
    #                   warm_start=False)

    # fitto il modello 
    model.fit(dati, annotazioni_ground_truth)

    # in alternativa posso fare tutto in una singola linea
    #model = LogisticRegression(solver='liblinear', random_state=0).fit(dati, annotazioni_ground_truth)
    print("Classi del modello: ", model.classes_)
    input("Modello generato, premi invio per continuare\n\n")

    # genero le probabilità e predizioni per ogni annotazione
    probabilita_annotazioni = model.predict_proba(dati)
    annotazioni_predette = model.predict(dati)


    # Plottiamo i risultati
    plt.plot(dati, annotazioni_predette, c='r', label='Modello')
    plt.scatter(dati, annotazioni_ground_truth, c='g', label='Data')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Dati')
    plt.ylabel('Annotazioni')
    plt.show()


    # generiamo le statistiche con la matrice di confusione e plottiamo i risultati
    score_accuratezza_media = model.score(dati, annotazioni_ground_truth)
    print("Accuratezza media = ", score_accuratezza_media, "% \n")
    matrice_di_confusione = confusion_matrix(annotazioni_ground_truth, annotazioni_predette)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(matrice_di_confusione)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predizione 0s', 'Predizione 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Dato vero 0s', 'Dato vero 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrice_di_confusione[i, j], ha='center', va='center', color='red')
    plt.show()

    print(classification_report(annotazioni_ground_truth, model.predict(dati)))
