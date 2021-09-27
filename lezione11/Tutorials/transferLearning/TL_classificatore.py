#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : MNIST                                                         |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Questo tutorial è pensato per mostrare il funzionamento del transfer learning 
usando la rete resnet18. 
Questo tutorial è rielaborato dai tutorial ufficiali di pytorch disponibili su
https://pytorch.org/tutorials/

"""
print(__doc__)

# importo librerie standard
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import copy

# importo librerie di torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR

def visulizzaBatch(_immagini, _annotazioni):
    """
    Semplice funzione per mostrare qualche immagine del nostro dataset

    Parametri: 
    ----------
    img(torchvision): immagini da visualizzare
    """
    _immagini = _immagini *0.22 + 0.45     # denormalizzazione
    npimg = _immagini.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    dataset_classes = ('formiche', 'ape')
    print('Predicted: ', ' '.join('%5s' % dataset_classes[_annotazioni.indices.cpu().numpy()[j]]
                                for j in range(4)))
        
    #titolo = "annotazioni: \n" + str(_annotazioni.indices.cpu().numpy() ) 
    titolo = "annotazioni: \n" + ' '.join('%5s' % dataset_classes[_annotazioni.indices.cpu().numpy()[j]]
                                for j in range(4)) 
    plt.title(titolo)
    plt.show()

def train(modello, device, train_loader, criterio, ottimizzatore, epoch):
    """
    Funzione di training della rete

    Parametri: 
    ----------
    argomenti
    modello (torch.nn): modello neurale definito nella classe, in questo caso MauroNet
    device (torch.device): può essere "cpu" o "gpu:idx" con idx indice della GPU in caso di sistemi multi-GPU
    train_loader (torch.dataloader): il data loader può essere importato (come in questo caso) o customizzato
    ottimizzatore (torch.optim): rappresenta l'ottimizzatore usato per il gradiente
    epoch (int): è l'epoca corrente, mi serve solo per stampare il risultato ogni intervallo di log
    """

    modello.train() # impostiamo il modello in modalità "train", questo è ereditato e permette di modificare i pesi

    for batch_idx, (data, annotazione) in enumerate(train_loader): # per tutti i dati
        
        data, annotazione = data.to(device), annotazione.to(device) # uso il data loader per caricare una coppia dato-annotazione
        ottimizzatore.zero_grad() # il gradiente deve essere azzerato ad ogni iterazione 
        output = modello(data)  # passo in avanti
        loss = criterio(output, annotazione) # calcolo la loss
        loss.backward()       # passo indietro
        ottimizzatore.step()      # aggiorno l'ottimizzatore 
        
        # Stampo delle statistiche ogni 1000 campioni (1000 è il mio intervallo di log)
        if batch_idx % 1000 == 0:
            print('Epoca di training: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            

def test(_modello, _device, _test_loader, _criterio, _visualizza_risultato=False ):
    """
    Funzione di testing della rete 

    Parametri: 
    ----------
    _modello (torch.nn): modello neurale definito nella classe, in questo caso MauroNet
    _device (torch.device): può essere "cpu" o "gpu:idx" con idx indice della GPU in caso di sistemi multi-GPU
    _test_loader (torch.dataloader): il data loader può essere importato (come in questo caso) o customizzato
    """
    _modello.eval() # impostiamo il modello in modalità "eval", questo è ereditato e impedisce di modificare i pesi
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = _modello(data)
            test_loss += _criterio(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if (_visualizza_risultato):
                visulizzaBatch(torchvision.utils.make_grid(data[:,:,:]), torch.max(output[:],dim=1) )

    test_loss /= len(_test_loader.dataset)

    print('\nTest set: Loss media: {:.4f}, Accuratezza globale: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(_test_loader.dataset),
        100. * correct / len(_test_loader.dataset)))



    
# entry point
if __name__ == '__main__':

    # Training settings
    
    # qualche parametro di training   
    learning_rate = 0.001
    train_batch_size = 8
    test_batch_size = 8
    max_epoch = 3
    gamma = 0.7
    
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = './data/hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    modello = models.resnet18(pretrained=True)
    num_ftrs = modello.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    modello.fc = nn.Linear(num_ftrs, 2)

    model_ft = modello.to(device)

    criterio = nn.CrossEntropyLoss().to('cuda:0')
    # Addestra tutti i parametri
    optimizer = optim.Adam(modello.parameters(), lr=learning_rate)
    # Addestra solo i livelli lineari "fc=fully connected" 
    #optimizer = optim.SGD(modello.fc.parameters(), lr=learning_rate, momentum=0.9)
    #optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    
    # inizio il training
    for epoch in range(1, max_epoch + 1):
        train(modello, device, dataloaders['train'], criterio, optimizer, epoch)
        test(modello, device, dataloaders['val'], criterio, True )
        scheduler.step()

    # per divertimento proviamo a visualizzare qualche risultato
    test(modello, device, dataloaders['val'], criterio, True )

    save_model=False
    if save_model:
        torch.save(modello.state_dict(), ".\TL_resnet18.pt")