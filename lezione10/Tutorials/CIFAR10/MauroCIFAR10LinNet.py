#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : MNIST                                                         |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

# importo librerie di torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definisco una classe con la mia rete neurale 
class MauroCIFAR10LinNet(nn.Module):
    """
    Classe per definire la rete neurale per risolvere un problema di classificazione logistica 
    in questo caso, riconoscimento dei numeri scritti a mano del dataset MNIST. 
    Vedi http://yann.lecun.com/exdb/mnist/ per maggiori dettagli. 

    Questa classe eredita i metodi di nn.Module, quindi le funzioni accessibili sono definite 
    nella classe madre
    """

    def __init__(self):
        """
        Reimplementazione del costrutture per definire i livelli neurali
        """
        super(MauroCIFAR10LinNet, self).__init__()
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        #NOTA: le immagini CIFAR10 sono costituite da caratteri registrati su 
        #      immagini della dimensione pari a 32x32x3 pixel = 3072 
        #      il task è quello di classificare in 10 classi in uscita, numeri da 0 a 9
        flatten_feature_num = 3
        flatten_tensor_size_x = 32
        flatten_tensor_size_y = 32
        self.flatten_size = flatten_feature_num * flatten_tensor_size_x * flatten_tensor_size_y
        
        self.fc1 = nn.Linear(in_features=self.flatten_size, out_features=256)  # fc -> sta per fully connected
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=128)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=64)
        self.fc4 = nn.Linear(in_features=self.fc3.out_features, out_features=10)

    def forward(self, x):
        """
        Reimplementazione del forward pass per definire il flusso di avanzamento 
        nella rete. 

        NOTA: non definiamo una funzione "backward" questo è automatico quando si 
              usa questa libreria, chiaramente per gestire il gradiente è possibile 
              utilizzare delle funzioni di backward personalizzate
        """
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.dropout1(x)
        x = self.fc3(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc4(x)
        output = x
        return output