#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : regessione lineare                                            |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Esempio di regressione lineare 
"""
print(__doc__)

# importo librerie standard, numpy per analisi matemtica e matplotlib per plottare i risultati
import numpy as np
import matplotlib.pyplot as plt

# importo sklearn libreria con molti pacchetti di analisi statistica e modelli di apprendimento
from sklearn.linear_model import LinearRegression

# genero dei vettori di esmpio che rappresentano il fatturato annuo
anno = np.array([2015, 2016, 2017, 2018, 2019, 2020]).reshape((-1, 1))
fatturato_mEuro = np.array([5500, 5600, 5450, 5650, 5800, 5750])

# stampo a schermo i vettori per vedere il loro contenuto
print(anno)
print(fatturato_mEuro)
input ("Dati generati, premi invio per continuare")

# genero il modello di regressione lineare
modello = LinearRegression()
modello.fit(anno, fatturato_mEuro)

# in alternativa possiamo istanziare e fittare in una unica linea
# modello = LinearRegression().fit(anno, fatturato_mEuro)

# Calcolo il coefficiente R^2 per valutare l'attendibilità del mio modello
#  𝑅^2≈1  Significa che le previsioni del modello sono attendibili
#  𝑅^2≈0  Significa che le previsioni del modello NON sono attendibili
R_quadro = modello.score(anno, fatturato_mEuro)
print('\nCoefficiente R^2: ', R_quadro, ' \n')

# stampo i parametri della retta di regressione
print('Coefficiente angolare della retta di regressione:', np.round(modello.coef_,2))
print('Quota della retta di regressione:', np.round(modello.intercept_,2) , '\n\n')

y_pred = modello.predict(anno)
print('Risposta predetta:', np.round(y_pred,1), sep='\n')


# build the state matrix
hessiano = np.zeros((len(anno),2))  
for i in range (1,len(anno)):
    hessiano[i,0] = anno[i]
    hessiano[i,1] = 1.0

# costruisce la stima dei parametri usando la matrice hessiana
parametri_MLS =  np.matmul(np.matmul(np.linalg.inv(np.matmul(hessiano.T, hessiano)),hessiano.T),fatturato_mEuro)
print(parametri_MLS)
input ("press enter")
#t = [0,x(size(x,1)-1)];
test_MLS = parametri_MLS[0]*anno + parametri_MLS[1]
print('Risposta predetta MLS:', np.round(test_MLS,1), sep='\n')

x_next = np.array([2021, 2022, 2023]).reshape((-1, 1))
y_next = modello.predict(x_next)
print("\nFatturato 2021-23 ", np.round(y_next,1))

# plot con il modello e i dati che sono stati usati per generarlo
plt.plot(anno, y_pred, c='r', label='Modello')
plt.scatter(anno, fatturato_mEuro, c='g', label='Data')
plt.scatter(anno, test_MLS, c='b', label='Data MLS')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Anno')
plt.ylabel('Fatturato in MEuro')
plt.show()

# plot con il modello, dati e previsioni
plt.plot(anno, y_pred, c='r', label='Modello')
plt.scatter(anno, fatturato_mEuro, c='g', label='Data')
plt.scatter(x_next, y_next, c='b', label='Predizione')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Anno')
plt.ylabel('Fatturato in MEuro')
plt.show()