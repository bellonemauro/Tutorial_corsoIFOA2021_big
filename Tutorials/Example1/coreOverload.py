#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial 1: monitoraggio risorse remote                                  |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Produce un carico per tutti cores nella CPU 

Source: https://docs.python.org/3/library/multiprocessing.html
"""

from multiprocessing import Pool
from multiprocessing import cpu_count

"""
Dummy function che produce un carico computazionale infinito
"""
def f(x):
    while True:   # ciclo infinito
        x*x       # funzione esponenziale

"""
Dummy function che produce un carico computazionale infinito e un overload di memoria
"""
def g(x):
    x = []
    while True:           # ciclo infinito
        x.append(1)       # funzione esponenziale

"""
Funzione main da eseguire
"""
if __name__ == '__main__':
    processes = cpu_count()                    # ritorna il numero di core della CPU
    print ("Utilizzo ", processes," cores\n")  # stampo a schermo
    pool = Pool(processes)                     # creo un vettore di processi grande quanto il mio numero di cores
    pool.map(f, range(processes))              # eseguo la funzione f per tutti i cores
    #pool.map(g, range(processes))              # eseguo la funzione g per tutti i cores