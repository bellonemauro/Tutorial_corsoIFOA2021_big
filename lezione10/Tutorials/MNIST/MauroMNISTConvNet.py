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
class MauroMNISTConvNet(nn.Module):
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
        # encoder 
        super(MauroMNISTConvNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # input 1x28x28
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5,5)) #stride=(1,2), padding=(2,0), dilation=1)
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=16, kernel_size=(5,5))
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=(3,3))

        # Fully connected context 
        flatten_feature_num = 32
        flatten_tensor_size_x = 1
        flatten_tensor_size_y = 1
        self.flatten_size = flatten_feature_num * flatten_tensor_size_x * flatten_tensor_size_y
        
        self.con_fc1 = nn.Linear(self.flatten_size, 30)

        self.fc1 = nn.Linear(in_features=self.flatten_size, out_features=25)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=20)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=10)

                                  

    def forward(self, x):
        """
        Reimplementazione del forward pass per definire il flusso di avanzamento 
        nella rete. 

        NOTA: non definiamo una funzione "backward" questo è automatico quando si 
              usa questa libreria, chiaramente per gestire il gradiente è possibile 
              utilizzare delle funzioni di backward personalizzate
        """

        #x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        