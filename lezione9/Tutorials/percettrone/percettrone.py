#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : Percettrone                                                   |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

# importo librerie standard
import numpy as np

class percettroneSemplice():
    """
    Implementazione classe percetrone semplice basato su numpy


    Esempio d'uso 
    --------
    >>> percettrone_semplice = percettroneSemplice(_num_input=num_input, _num_iterazioni = 10000, _learning_rate =0.001) 
    >>> percettrone_semplice.ottimizza(dati_train, annotazioni_train)
    >>> annotazioni_pred = percettrone_semplice.predict(dati)

    """

    def __init__(self, _num_input, _num_iterazioni, _learning_rate):
        """
        Costruttore dell'oggetto percettrone.
        Inizia le variabili membro della classe e include l'inizializzazione random dei pesi

        Parametri:
        ----------
            _num_input (int): Numero di input
            _num_iterazioni (int): numero di iterazioni
            _learning_rate (int): learning rate
        """
        print("Istanzio la classe percettrone  ")
        self.num_input = _num_input
        self.learning_rate = _learning_rate
        self.num_iterazioni = _num_iterazioni
        self.costo_corrente = np.inf
        self.param_dict = []
        self.grad_dict = []
        self.costi = []

        self.pesi, self.bias = self.inizializzazione_pesi(self.num_input)
        print("I pesi random generati sono: ", self.pesi)
        print("I bias random generati sono: ", self.bias)

    # Definisco la funzione sigmoide
    def sigmoide(self, _a):
        """
        Semplice implementazione della funzione di attivazione sigmoide 

        Parametri: 
        ----------
        _a (float): valore nel quale calcolare la funzione di attivazione 

        Ritorna:
        ----------
            valore della sigmoide in _a
        """
        return 1/(1 + np.exp(-_a))

    
    def inizializzazione_pesi(self, _dim):
        """
        Funzione di inizializzazione random dei pesi 

        Parametri: 
        ----------
        _dim (int): numero di pesi da inizializzare 

        Ritorna:
        ----------
        pesi (array[_dim]): array di valori float della dimensione in input _dim
        bias (float): singolo valore float 
        """
        # in questo caso stiamo usando una distribuzione normale per generare i pesi iniziali
        pesi = np.random.normal((_dim,1)) 
        bias = 0  # NOTA: tutti i bias sono inizializzati a 0
        return pesi, bias
 

    def propaga(self, _input, _output):
        """
        Funzione di propagazione dell'errore, esegue i passi avanti e indietro
        per ri-settare i pesi del percettrone ad ogni iterazione

        Parametri: 
        ----------
        _input (array[float]): dati sui quali effettuare il training
        _annotazioni (array[float]): annotazioni relative ai dati in input
        
        """
        # propagazione in avanti
        m = _input.shape[1]
        attivazione = self.sigmoide(np.dot(self.pesi.T, _input) + self.bias)

        # calcolo la funzione di costo rappresentata da una cross entropy
        self.costo_corrente = (- 1 / m) * np.sum(_output * np.log(attivazione) + (1 - _output) * (np.log(1 - attivazione)))
        # FINE propagazione in avanti
        
        # Propagazione all'indietro
        d_pesi = (1/m)*np.dot(_input, (attivazione-_output).T)
        d_bias = (1/m)*np.sum(attivazione-_output)
        # FINE: propagazione all'indietro
        
        # controllo che le dimensioni e le forme siano corrette
        assert(d_pesi.shape == self.pesi.shape)
        assert(d_bias.dtype == float)
        self.costo_corrente = np.squeeze(self.costo_corrente)
        assert(self.costo_corrente.shape == ())
        
        # struttura dati che salva le derivate dei gradienti 
        self.grad_dict = {"d_p":d_pesi , "d_b":d_bias}
        
  
    def ottimizza(self, _input, _annotazioni):
        """
        Ottimizzatore del percettrone basato sul gradient descent. 
        Questa funzione non ritorna nulla, i pesi sono aggiornati nelle variabili membro 
        della classe percettrone i cui valori sono accessibili tramite self.pesi, self.bias

        Parametri: 
        ----------
        _input (array[float]): dati sui quali effettuare il training
        _annotazioni (array[float]): annotazioni relative ai dati in input
        """
        self.costi = [] # deve essere azzerata ogni volta che inizio l'ottimizzatore
        for i in range(self.num_iterazioni):
            
            self.propaga(_input, _annotazioni)
            d_pesi = self.grad_dict["d_p"]
            d_bias = self.grad_dict["d_b"]
            
            #Gradient Descent - equazione destrtturata
            self.pesi = self.pesi - self.learning_rate * d_pesi
            self.bias = self.bias - self.learning_rate * d_bias
            #Gradient Descent
            
            if i % 1000 == 0:   # stampo il costo solo ogni 1000 iterazioni
                self.costi.append(self.costo_corrente)
                print ("Costo dopo %i iterazioni = %f" % (i, self.costo_corrente))
                
        self.param_dict = {"p":self.pesi, "b":self.bias}
        self.grad_dict = {"d_p":d_pesi, "d_b":d_bias}
        
        

    # Funzione di predizione (inferenza)
    def predict(self, _input):
        """
        Funzione di predizione dato, assume che il modello sia già generato, 
        la dimensione di _input deve avere un numero di colonne pari al numero di pesi generati,
        mentre non c'è limite al numero di righe

        Parametri: 
        ----------
        _input (array[float]): dati sui quali effettuare l'inferenza

        Ritorna:
        ----------
        output(array[float]): dati in uscita dal percettrone, singolo uscita per ogni ingresso
        """
        m = _input.shape[1]
        output_pred = np.zeros((1,m))
        self.pesi = self.pesi.reshape(_input.shape[0], 1)
        
        attivazione = self.sigmoide(np.dot(self.pesi.T, _input)+ self.bias)
        
        for i in range(m):
            output_pred[0,i] = 1 if attivazione[0,i] > 0.5 else 0  #il neurone è attivo solo se la sigmoide è > 0.5
            
        assert(output_pred.shape == (1, m))
            
        return output_pred
