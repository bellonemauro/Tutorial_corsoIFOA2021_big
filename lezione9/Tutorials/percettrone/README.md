# Descrizione Esempio 

Tutorial sviluppato per comprendere il funzionamento di un semplice percettrone in python.

Il codice è composto da una classe percettrone, ed uno script che istanza l'oggetto nella classe ed effettua il training usando il gradient descent. 

## Modello del percettrone implementato 

Il modello implementato è un percettrone con n-ingressi e funzione di attivazione sigmoide


<img src="https://github.com/bellonemauro/Tutorial_corsoIFOA2021_big/blob/main/lezione9/Tutorials/percettrone/percettrone.png"  width="370" height="135" />





## Plot dati in ingresso 

Il modello ha in ingresso dei dati autogenerati in maniera causale con due centroidi

<img src="https://github.com/bellonemauro/Tutorial_corsoIFOA2021_big/blob/main/lezione9/Tutorials/percettrone/screen_result.png"  width="643" height="548" />


## Plot classi in uscita con training e test

Il modello in uscita, avendo un solo neurone, può semplicemente stimare due pesi $w_i$ e tracciare una retta tra le classi. 

<img src="https://github.com/bellonemauro/Tutorial_corsoIFOA2021_big/blob/main/lezione9/Tutorials/percettrone/screen_result_2.png"  width="643" height="548" />


## Andamento della funzione di loss

La funzione di loss è decrescete come ci si aspetta nel training di una rete neurale

<img src="https://github.com/bellonemauro/Tutorial_corsoIFOA2021_big/blob/main/lezione9/Tutorials/percettrone/screen_result_3.png"  width="643" height="548" />
