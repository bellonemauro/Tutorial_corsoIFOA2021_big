#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : Regole associative                                           |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 


"""
Questo tutorial è pensato per mostrare il funzionamento delle regole associative 
e dei principali parametri coinvolti nel loro studio. 

"""



# importo librerie utili 
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules
import time




# Creo il mio dataset fatto semplicemente come elenco di transazioni e oggetti acquistati
# in questo dataset ogni vettore indica la lista della spesa per un cliente nel seguente formato: 
#
# | cliente 1 | [lista oggetti]
# | cliente 2 | [lista oggetti]
#
# | cliente n | [lista oggetti]   ---> le liste possono avere dimensioni diverse
dataset = [['Latte', 'Cipolla', 'Arachidi', 'Cereali', 'Uova', 'Yogurt'],
           ['Basilico', 'Cipolla', 'Arachidi', 'Cereali', 'Uova', 'Yogurt'],
           ['Latte', 'Mele', 'Cereali', 'Uova'],
           ['Latte', 'Cavallo', 'Pane', 'Cereali', 'Yogurt'],
           ['Pane', 'Cipolla', 'Cipolla', 'Cereali', 'Gelato', 'Uova']]

minimo_supporto = 0.6
minima_confidenza = 0.8
minimo_lift = 1.2
minimo_numero_di_antecedenti = 0

# instanziamo l'oggetto di codifica delle transazioni
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
data_frame = pd.DataFrame(te_ary, columns=te.columns_)

# eseguiamo l'algoritmo di ricerca delle associazioni apriori
inizio = time.time()
oggetti_frequenti_ap = apriori(data_frame, min_support=minimo_supporto, use_colnames=True)
fine = time.time()
print("Tempo di generazione delle degli insiemi di oggetti frequenti con apriori pari a ", fine - inizio, " sec")

### A scopo didattico possiamo sostituire la linea precedente con una delle seguenti e testare fpgrowth:
inizio = time.time()
oggetti_frequenti = fpgrowth(data_frame, min_support=minimo_supporto, use_colnames=True)

fine = time.time()
print("Tempo di generazione delle degli insiemi di oggetti frequenti con fpgrowth pari a ", fine - inizio, " sec")
input (" premi un tasto ")
# stampiamo le frequenze
print(oggetti_frequenti)

# generiamo le regole di associazione usando diverse metriche 
regole_confidenza = association_rules(oggetti_frequenti, metric="confidence", min_threshold=minima_confidenza)
print(regole_confidenza)
input("Associazioni usando la confidenza come metrica generate - premi un tasto per continuare\n\n")


regole_lift = association_rules(oggetti_frequenti, metric="lift", min_threshold=minimo_lift)
print(regole_lift)
input("Associazioni usando la metrica lift generate - premi un tasto per continuare\n\n")

# aggiungiamo una linea nella quale annotiamo il numero di antecedenti, quindi il dataframe regole_XX 
# avrà una colonna aggiuntiva chiamata "antecedent_len" contenente len(x) quindi la lunghezza del vettore antecedenze
regole_confidenza["antecedent_len"] = regole_confidenza["antecedents"].apply(lambda x: len(x))
print(regole_confidenza)
input("Lunghezza delle antecedenze aggiunta - premi un tasto per continuare\n\n")

#facciamo la stessa cosa sulle regole lift 
regole_lift["antecedent_len"] = regole_lift["antecedents"].apply(lambda x: len(x))
print(regole_lift)
input("Lunghezza delle antecedenze aggiunta - premi un tasto per continuare\n\n")

# filtriamo le regole generate secondo parametri diversi per ottenere le righe di interesse
regole_migliori_conf = regole_confidenza[ (regole_confidenza['antecedent_len'] >= minimo_numero_di_antecedenti) &
       (regole_confidenza['confidence'] >= minima_confidenza) &
       (regole_confidenza['lift'] > minimo_lift) ]
print(regole_migliori_conf)
input("Limite sulle antecedenze, confidenza e lift - premi un tasto per continuare\n\n")

regole_migliori_lift = regole_lift[ (regole_lift['antecedent_len'] >= minimo_numero_di_antecedenti) &
       (regole_lift['confidence'] >= minima_confidenza) &
       (regole_lift['lift'] > minimo_lift) ]
print(regole_migliori_lift)
input("Limite sulle antecedenze, confidenza e lift - premi un tasto per continuare\n\n")

