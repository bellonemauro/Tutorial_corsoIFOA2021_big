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
class MauroMNISTLinNet(nn.Module):
    """
    Classe per definire la rete neurale per risolvere un problema di classificazione logistica 
    in questo caso, riconoscimento dei numeri scritti a mano del dataset MNIST. 
    Vedi http://yann.lecun.com/exdb/mnist/ per maggiori dettagli. 

    Questa classe eredita i metodi di nn.Module, quindi le funzioni accessibili sono definite 
    nella classe madre
    """

    def __init__(self, _out_classes):
        """
        Reimplementazione del costrutture per definire i livelli neurali
        """
        super(MauroMNISTLinNet, self).__init__()
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

        #NOTA: le immagini MNIST sono costituite da caratteri registrati su 
        #      immagini della dimensione pari a 28x28 pixel = 784 
        #      il task è quello di classificare in 10 classi in uscita, numeri da 0 a 9
        flatten_feature_num = 1
        flatten_tensor_size_x = 28
        flatten_tensor_size_y = 28
        self.flatten_size = flatten_feature_num * flatten_tensor_size_x * flatten_tensor_size_y #784
        
        self.input_layer = nn.Linear(in_features=self.flatten_size, out_features=256)  # fc -> sta per fully connected
        self.hidden_layer1 = nn.Linear(in_features=self.input_layer.out_features, out_features=128)
        self.hidden_layer2 = nn.Linear(in_features=self.hidden_layer1.out_features, out_features=64)
        self.out_layer = nn.Linear(in_features=self.hidden_layer2.out_features, out_features=_out_classes)

    def forward(self, x):
        """
        Reimplementazione del forward pass per definire il flusso di avanzamento 
        nella rete. 

        NOTA: non definiamo una funzione "backward" questo è automatico quando si 
              usa questa libreria, chiaramente per gestire il gradiente è possibile 
              utilizzare delle funzioni di backward personalizzate
        """
        x = torch.flatten(x, 1)

        x = self.input_layer(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden_layer1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden_layer2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.out_layer(x)
        x = F.log_softmax(x, dim=1)
        output = x
        return output