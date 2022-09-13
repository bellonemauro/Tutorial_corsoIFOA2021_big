#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : CIFAR                                                         |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

# importo librerie di torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definisco una classe con la mia rete neurale 
class MauroCIFAR10ConvNet(nn.Module):
    """
    Classe per definire la rete neurale per risolvere un problema di classificazione logistica 

    Questa classe eredita i metodi di nn.Module, quindi le funzioni accessibili sono definite 
    nella classe madre
    """

    def __init__(self):
        """
        Reimplementazione del costrutture per definire i livelli neurali
        """
        # encoder 
        super(MauroCIFAR10ConvNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5,5), padding=(2,2)) #stride=(1,2), padding=(2,0), dilation=1)
        self.bn1 = nn.BatchNorm2d(num_features = self.conv1.out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64, kernel_size=(5,5), padding=(2,2))
        self.bn2 = nn.BatchNorm2d(num_features = self.conv2.out_channels)
        
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=128, kernel_size=(5,5), padding=(2,2))
        self.bn3 = nn.BatchNorm2d(num_features = self.conv3.out_channels)
        
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128, kernel_size=(5,5), padding=(2,2))
        self.bn4 = nn.BatchNorm2d(num_features = self.conv4.out_channels)
        
        self.conv5 = nn.Conv2d(in_channels=self.conv4.out_channels, out_channels=128, kernel_size=(3,3), padding=(2,2))
        self.bn5 = nn.BatchNorm2d(num_features = self.conv5.out_channels)

        # Fully connected context 
        flatten_feature_num = self.conv3.out_channels
        flatten_tensor_size_x = 2
        flatten_tensor_size_y = 2
        self.flatten_size = flatten_feature_num * flatten_tensor_size_x * flatten_tensor_size_y
        self.dropout = nn.Dropout(p=0.25)
        #self.con_fc1 = nn.Linear(self.flatten_size, 100)

        self.fc1 = nn.Linear(in_features=self.flatten_size, out_features=64)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=32)
        self.fc3 = nn.Linear(in_features=self.fc2.out_features, out_features=10)

                                  

    def forward(self, x):
        """
        Reimplementazione del forward pass per definire il flusso di avanzamento 
        nella rete. 

        NOTA: non definiamo una funzione "backward" questo è automatico quando si 
              usa questa libreria, chiaramente per gestire il gradiente è possibile 
              utilizzare delle funzioni di backward personalizzate
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool(x)
        #x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.elu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.elu(x)
        x = self.pool(x)
        
        x = x.view(-1, self.flatten_size)
        x = self.dropout(F.elu(self.fc1(x)))
        x = self.dropout(F.elu(self.fc2(x)))
        x = self.fc3(x)
        return x
        
