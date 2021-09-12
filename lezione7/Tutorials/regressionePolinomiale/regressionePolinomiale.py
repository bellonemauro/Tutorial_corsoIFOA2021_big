#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : regessione polinomiale                                        |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Esempio di regressione polinomiale
"""
print(__doc__)

# importiamo librerie di base, matematica e plotting
import numpy as np
import matplotlib.pyplot as plt

# importiamo gli strumenti che ci servono da sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# genero l'asse delle ascisse come numeri da 10 a 100 distanziati di 1.0 unità
dati_x = np.linspace(1.0, 10.0, 100)[:, np.newaxis]

# genero le misure della grandezza di interesse secondo una funzione sin(x)+alfa*x^2+beta*random()
# questo fornirà delle misure affette da un errore casuale 
alfa = 2.0 #10.0
beta = 0.1
gamma = 0.4
grado_funzione = 2  #3#2
misure_y = alfa*np.sin(dati_x) + beta*np.power(dati_x,grado_funzione) + gamma*np.random.randn(100,1)
dati_x /= np.max(dati_x)


# Proviamo a plottare polinomiali di grado da 1 a 10
for grado_polinomiale in range (1,10):

    x_ = PolynomialFeatures(degree=grado_polinomiale, include_bias=True).fit_transform(dati_x)
    model = LinearRegression(fit_intercept=False).fit(x_, misure_y)

    # in algernativa possiamo destrutturare le linee come segue
    #transformer = PolynomialFeatures(degree=grado_polinomiale, include_bias=True)
    #transformer.fit(dati_x)
    #x_ = transformer.transform(dati_x)
    #model = LinearRegression(fit_intercept=False).fit(x_, misure_y)

    # Calcolo il coefficiente R^2 per valutare l'attendibilità del mio modello
    #  𝑅^2≈1  Significa che le previsioni del modello sono attendibili
    #  𝑅^2≈0  Significa che le previsioni del modello NON sono attendibili
    R_quadro = model.score(x_, misure_y)
    print('Coefficiente di determinazione R^2:', R_quadro)
    print('\nCoefficienti curva di regressione:', model.coef_)
    print('Quota della rett di regressione:', model.intercept_ , "\n\n")
    
    # genero la curva del modello 
    y_modello = model.predict(x_)
    #print('Risposta del modello:', y_modello, sep='\n')

    # provo a generare dei dati non visti in precedenza
    x_next = np.array([0.15, 0.35, 0.43, 0.6, 0.72, 0.95]).reshape((-1, 1))
    x_p_next = PolynomialFeatures(degree=grado_polinomiale, include_bias=True).fit_transform(x_next)
    y_next = model.predict(x_p_next)

    # genero il plot dei dati e della curva polinomiale approssimante
    plt.plot(dati_x, y_modello, c='r', label='Modello') # plotta una linea
    plt.scatter(dati_x, misure_y, c='g', label='Data') # grafico a punti 
    plt.scatter(x_next, y_next, c='b', label='Predizione')
    titolo = "Polinomiale grado " + str(grado_polinomiale)
    titolo += ", coefficiente di determinazione R^2 = " + str(np.round(R_quadro,3))
    plt.title(titolo)
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

