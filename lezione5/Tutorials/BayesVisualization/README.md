# Guida

Questa applicazione visualizza facilmente gli effetti di un test diagnostico 
su una popolazione usando la [legge di Bayes](https://en.wikipedia.org/wiki/Bayes%27_theorem). 

Dati due eventi $A$ e $B$ la legge di Bayes ci dice che la probabilità condizionata 
di verificazione dell'evento $A$, dato che l'evento $B$ si è verificato, è calcolabile come: 

$P(A|B) = P(B|A) * P(A) / P(B)$

Dove: 
- $P(A)$ è la probabilità di verificazione dell'evento $A$
- $P(B)$ è la probabilità di verificazione dell'evento $B$
- $P(B|A)$ è la probabilità di verificazione dell'evento $B$ dato l'evento $A$ verificato

Un esempio analogo è disponibile su wikipedia
per la determinazione della probabilità di essere positivi in un
[drug test](https://en.wikipedia.org/wiki/Bayes%27_theorem#Drug_testing)
data l'esito positivo del test.

## Problema
Supponiamo di eseguire un test diagnostico su una popolazione di N soggetti per la ricerca di 
una patologia (es. Maurite). La maurite ha una prevalenza nella popolazione 
regolabile usando l'indicatore -prevalenza- che rappresenta la percentuale di popolazione che 
effettivamente ha la maurite (valore di default 10%). 

Gli scienziati hanno sviluppato un test diagnostico che riesce a rilevare la maurite. 
La probabilità di risultare positivo al test diagnostico per un soggetto avente la maurite 
è pari al valore di sensibilità del test (valore di default 90%). 
La probabilità di risultare negativo al test diagnostico per un soggetto sano è pari al 
valore di specificità del test (valore di default pari al 95%).

Siamo interessati a sapere qual è la probabilità, per un soggetto che ha ricevuto un test 
con esito positivo, di essere effettivamente positivo. 

Il risultato non è affatto banale e fortemente controintuitivo. 
Questa dashboard calcola le probabilità condizionate di essere positivo avendo avuto ricevuto 
un test con esito positivo, e la probabilità di essere negativo avendo ricevuto un test con esito negativo. 

## Risultato analitico

Siano $+$ e $–$ gli eventi risultati dall'esito del test diagnostico che può essere positivo o negativo, 
quindi $D$ e $𝐷^𝐶$ sono gli eventi the risultano dal soggetto di aver contratto o no la maurite. 

La sensibilità è la probabilità che il test diagnostico sia positivo dato che il soggetto è effettivamente positivo $𝑃(+|𝐷)$. 

La specificità è la probabilità che il test sia negativo dato che il soggetto non ha effettivamente la maurite $𝑃(−|𝐷^𝐶)$.

Se un soggetto ha ricevuto un test con esito positivo è interessato alla probabilità che $𝑃(𝐷|+)$, 
cioè la probabilità di avere un test positivo e di aver effettivamente contratto la malattia. 

Se hai avuto un test negativo sei interessato alla probabilità che $𝑃(𝐷^𝐶 |−)$, 
cioè la probabilità di aver avuto un test negativo e di non contratto la malattia. 

$𝑃(𝐷|+) = 𝑃(+|𝐷) 𝑃(+) / 𝑃(𝐷) = 𝑃(+|𝐷)𝑃(+) / (𝑃(+|𝐷)𝑃(+)+𝑃(𝐷│−)𝑃(−))$

## Risultato numerico

Eseguire il codice 
https://share.streamlit.io/bellonemauro/tutorial_corsoifoa2021_big/main/lezione5/Tutorials/BayesVisualization/bayesVisualization.py

