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
neurale addestrandola sul famoso dataset CIFAR10. 
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
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR

# importo il mio file con la rete personalizzata
from MauroCIFAR10LinNet import *
from MauroCIFAR10ConvNet import *

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
    dataset_classes = ('aereo', 'automobile', 'uccello', 'gatto',
            'cervo', 'cane', 'rana', 'cavallo', 'nave', 'camion')
    #print('Predicted: ', ' '.join('%5s' % dataset_classes[_annotazioni.indices.cpu().numpy()[j]]
    #                            for j in range(8)))
    #titolo = "annotazioni: \n" + str(_annotazioni.indices.cpu().numpy() ) 
    titolo = "annotazioni: \n" + ' '.join('%5s' % dataset_classes[_annotazioni.indices.cpu().numpy()[j]]
                                for j in range(8)) 
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
            

def test(modello, device, test_loader, criterio, visualizza_risultato=False ):
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
    classi_dataset = ('aereo', 'automobile', 'uccello', 'gatto',
            'cervo', 'cane', 'rana', 'cavallo', 'nave', 'camion')
    predizioni_corrette = {nome_classe: 0 for nome_classe in classi_dataset}
    tot_pred = {nome_classe: 0 for nome_classe in classi_dataset}

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = modello(data)
            test_loss += criterio(output, targets).item()  # sum up batch loss
            preds = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            corrette += preds.eq(targets.view_as(preds)).sum().item()
            
            for target, predizione in zip(targets, preds):
                if target == predizione:
                    predizioni_corrette[classi_dataset[target]] += 1
                tot_pred[classi_dataset[target]] += 1
            
            if (visualizza_risultato):
                visulizzaBatch(torchvision.utils.make_grid(data[0:8,:,:]), torch.max(output[0:8],dim=1) )

    test_loss /= (len(test_loader.dataset)/test_loader.batch_size)

    for nome_classe, correct_count in predizioni_corrette.items():
        accuracy = 100 * float(correct_count) / tot_pred[nome_classe]
        print("L'accuratezza per la classe {:5s} è: {:.1f} %".format(nome_classe, 
                                                   accuracy))
    print('\nTest set: Loss media: {:.4f}, Accuratezza globale: {}/{} ({:.0f}%)\n'.format(
        test_loss, corrette, len(test_loader.dataset),
        100. * corrette / len(test_loader.dataset)))
    
    


    
# entry point
if __name__ == '__main__':

    # Training settings
    
    # qualche parametro di training   
    learning_rate = 0.001
    train_batch_size = 16
    test_batch_size = 16
    max_epoch = 30
    gamma = 0.7
    passo_di_scheduling=10
    
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")


    trasformazioni = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])
    

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=trasformazioni)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=trasformazioni)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                            shuffle=False, num_workers=0)

    #modello = MauroCIFAR10LinNet().to(device)
    modello = MauroCIFAR10ConvNet().to(device)
    
    pytorch_total_params = sum(p.numel() for p in modello.parameters() if p.requires_grad)       
    print("Modello istanziato il numero totale di parametri allenabili è : ", pytorch_total_params)
    input("Premi invio per avviare il traning ")

    criterio = nn.CrossEntropyLoss().to(device)
    #ottimizzatore = optim.SGD(modello.parameters(), lr=learning_rate, momentum=0.9)
    ottimizzatore = optim.Adam(modello.parameters(), lr=learning_rate)
    #ottimizzatore = optim.Adadelta(modello.parameters(), lr=learning_rate)
    scheduler = StepLR(ottimizzatore, step_size=passo_di_scheduling, gamma=gamma)
   
    # inizio il training
    for epoca in range(1, max_epoch + 1):
        train(modello, device, train_loader, criterio, ottimizzatore, epoca)
        test(modello, device, test_loader, criterio,  False )
        scheduler.step()

    # per divertimento proviamo a visualizzare qualche risultato
    test(modello, device, test_loader, criterio, True )

    save_model=False
    if save_model:
        torch.save(modello.state_dict(), ".\cifar_mauroNet.pt")