#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : LSTM                                                         |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 
import torch
import torch.nn as nn


# classe di training con il modulo LSTM 
class MauroLSTM(nn.Module):
    """
    Classe per definire la rete neurale per risolvere un problema di regressione  
    tramite le reti LSTM. 
    Questa classe eredita i metodi di nn.Module, quindi le funzioni accessibili sono definite 
    nella classe madre
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MauroLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Reimplementazione del forward pass per definire il flusso di avanzamento 
        nella rete. 
        NOTA: non definiamo una funzione "backward" questo è automatico quando si 
              usa questa libreria, chiaramente per gestire il gradiente è possibile 
              utilizzare delle funzioni di backward personalizzate
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
