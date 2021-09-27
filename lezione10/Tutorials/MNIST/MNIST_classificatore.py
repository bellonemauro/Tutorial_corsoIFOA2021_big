#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : MNIST                                                         |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Questo tutorial è pensato per mostrare il funzionamento di una semplice rete 
neurale addestrandola sul famoso dataset MNIST. 
Questo tutorial è rielaborato dai tutorial ufficiali di pytorch disponibili su
https://pytorch.org/tutorials/

"""
print(__doc__)

# importo librerie standard
import numpy as np
import matplotlib.pyplot as plt

# importo librerie di torch
import argparse
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

# importo il mio file con la rete personalizzata
from MauroMNISTLinNet import *
from MauroMNISTConvNet import *

def visulizzaBatch(_immagini, _annotazioni):
    """
    Semplice funzione per mostrare qualche immagine del nostro dataset

    Parametri: 
    ----------
    img(torchvision): immagini da visualizzare
    """
    _immagini = _immagini * 0.3081 + 0.1307     # denormalizzazione
    npimg = _immagini.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    titolo = "annotazioni: \n" + str(_annotazioni.indices.cpu().numpy()) 
    plt.title(titolo)
    plt.show()

def training(modello, device, train_loader, criterio, ottimizzatore, epoch):
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
            

def test(modello, device, test_loader, criterio, visualizza_risultato=False):
    """
    Funzione di testing della rete 

    Parametri: 
    ----------
    modello (torch.nn): modello neurale definito nella classe, in questo caso MauroNet
    device (torch.device): può essere "cpu" o "gpu:idx" con idx indice della GPU in caso di sistemi multi-GPU
    test_loader (torch.dataloader): il data loader può essere importato (come in questo caso) o customizzato
    """
    modello.eval() # impostiamo il modello in modalità "eval", questo è ereditato e impedisce di modificare i pesi
    test_loss = 0
    corrette = 0

    classi_dataset = ('0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9')
    predizioni_corrette = {nome_classe: 0 for nome_classe in classi_dataset}
    tot_pred = {nome_classe: 0 for nome_classe in classi_dataset}

    # toch.no_grad significa che torch non calcolerà il gradiente della funzione
    with torch.no_grad():
        for data, targets in test_loader:   # per tutti i dati

            data, targets = data.to(device), targets.to(device) # prendi un esempio
            output = modello(data) # passo in avanti
            
            test_loss += criterio(output, targets).item()  # sum up batch loss
            preds = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            corrette += preds.eq(targets.view_as(preds)).sum().item()
            
            for target, predizione in zip(targets, preds):
                if target == predizione:
                    predizioni_corrette[classi_dataset[target]] += 1
                tot_pred[classi_dataset[target]] += 1
            
            if (visualizza_risultato):
                visulizzaBatch(torchvision.utils.make_grid(data[0:8,:,:]), torch.max(output[0:8],dim=1))

    test_loss /= (len(test_loader.dataset)/test_loader.batch_size)

    print('\nTest set: Loss media: {:.4f}, Accuratezza globale: {}/{} ({:.0f}%)\n'.format(
        test_loss, corrette, len(test_loader.dataset),
        100. * corrette / len(test_loader.dataset)))



    
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

    device = torch.device("cuda:0" if use_cuda else "cpu")

    trasformazioni = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307), (0.3081))
        ])

    dataset_training = datasets.MNIST('./data', train=True, download=True,
                       transform = trasformazioni)
    dataset_test = datasets.MNIST('./data', train=False,
                       transform = trasformazioni)
    train_loader = torch.utils.data.DataLoader(dataset_training, batch_size=train_batch_size,
                                            shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=test_batch_size,
                                            shuffle=False, num_workers=0)

    #modello = MauroMNISTLinNet(10).to(device)
    modello = MauroMNISTConvNet().to(device)
    pytorch_total_params = sum(p.numel() for p in modello.parameters() if p.requires_grad)       
    print("Modello istanziato il numero totale di parametri allenabili è : ", pytorch_total_params)
    input("Premi invio per avviare il traning ")
    
    criterio = nn.CrossEntropyLoss().to(device)
    #criterio = nn.NLLLoss().to(device)
    #ottimizzatore = optim.SGD(modello.parameters(), lr=learning_rate, momentum=0.9)
    ottimizzatore = optim.Adam(modello.parameters(), lr=learning_rate)
    #ottimizzatore = optim.Adadelta(model.parameters(), lr=learning_rate)
        
    # inizio il training
    for epoca in range(1, max_epoch + 1):
        training(modello, device, train_loader, criterio, ottimizzatore, epoca)
        test(modello, device, test_loader, criterio, False)
        

    # per divertimento proviamo a visualizzare qualche risultato
    test(modello, device, test_loader, criterio, True)

    save_model=False
    if save_model:
        torch.save(modello.state_dict(), ".\mnist_mauroNet.pt")