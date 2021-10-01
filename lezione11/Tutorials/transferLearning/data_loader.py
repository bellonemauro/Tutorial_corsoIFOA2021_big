#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : Transfer learning                                             |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

from __future__ import print_function, division
import os
from os import walk

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform

class myDataLoader(Dataset):
    """Data loader di esempio per immagini 

        This dataset is built to load patient data from a dataset of ultrasonic CSV frames. 
        
        

        Struttura della cartella attesa : 
        ./data/hymenoptera_MAURO/nome_immagine.ext 
        file_info.csv ----> file che contene i metadata per il data loader 

        La struttura del file csv di caricamento è atteso nella seguente forma:
        
            file_name,Class,Batch_index
            0013035.jpg,ants,1
            1030023514_aad5c608f9.jpg,ants,1
            etc.

        where:
            File_name ---> nome dell'immagine
            Class ---> annotazione
            Batch_index ---> indice del batch per cross validazione
                    

        l'uscita della reimplementazione di _getItem_ è composta come segue: 
            esempio ['immagine','annotazione']
        
    """

    def __init__(self, csv_file, root_dir, phase = 'train', num_of_classes = 2, val_batch_idx = 2, transform=None):
        """
        Args:
            csv_file (string): Percorso al file contenente l'indice dei frames e le relative informazioni
            root_dir (string): Percorso ai file.
            phase (string): deve essere nel range [train, val, test] 
            transform (callable, optional): Trasformazioni da applicare ad ogni esempio
        """
        self.lista_dei_dati = pd.read_csv(csv_file)
        self.phase = phase
        
        if (self.phase == 'train'):
            data_filter = (self.lista_dei_dati['Batch_index']!=val_batch_idx)
            self.lista_dei_dati = self.lista_dei_dati[data_filter]

        elif (self.phase == 'val' or self.phase == 'test'):
            data_filter = (self.lista_dei_dati['Batch_index']==val_batch_idx)
            self.lista_dei_dati = self.lista_dei_dati[data_filter]
        # else it does nothing so no filter is applied
            
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = [] 
        self.dir_list = [] 
        
        self.num_of_classes = num_of_classes
        self.classes = ['ants', 'bees']
            
        for (dirpath_in, dirnames_in, filenames_in) in walk(self.root_dir):
            self.file_list.extend(filenames_in)
            break

    def __len__(self):
        return len(self.lista_dei_dati)
       
    
        
    """
    Ridefinizione di Dataset.__getitem__
    """
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frame_name = self.lista_dei_dati.iloc[idx,0]
        annotazione = self.lista_dei_dati.iloc[idx, 1]

        if (annotazione == self.classes[0]):
            annotazione = np.asarray(0)
        elif (annotazione== self.classes[1] ):
            annotazione = np.asarray(1)

        immagine = io.imread(os.path.join(self.root_dir, frame_name))
        immagine = immagine/255 #normalizza in 0-1
        esempio = {'immagine': immagine, 'annotazione': annotazione}

        if self.transform:
            esempio = self.transform(esempio)

        return esempio


class MyNormalize(object):
    """Normalizza l'immagine a valori di media e dimensione standard fissati.

    Args:
        Fttore di normalizzazione media e deviazione standard (tuple ): .
    """

    def __init__(self, mean, std):
        #assert isinstance(mean, (float, [float]))
        #assert isinstance(std, (float, [float]))
        self.mean = mean 
        self.std = std

    def __call__(self, esempio):
        immagine, annotazione, = esempio['immagine'], esempio['annotazione']
        immagine = transforms.functional.normalize(torch.as_tensor(immagine), 
                                torch.as_tensor(self.mean), 
                                torch.as_tensor(self.std))  
        
        return {'immagine': immagine,
                'annotazione': annotazione }

class MyRescale(object):
    """Riscala l'immagine ad una dimensione data.

    Args:
        output_size (tuple or int): La dimensione in output desiderata. 
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, esempio):
        image, annotazione = esempio['immagine'], esempio['annotazione']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'immagine': img,
                'annotazione': annotazione }

class MyToTensor(object):
    """Converte ndarrays in un tensore esempio."""

    def __call__(self, esempio):
        immagine, annotazione = esempio['immagine'], esempio['annotazione']
        # scambiamo i colori degli assi in quanto: 
        # una immagine in numpy è rappresentata come: H x W x C
        # mentre su torch una immagine è rappresentata come: C X H X W
        immagine = immagine.transpose((2, 0, 1))
        
        return {'immagine': torch.FloatTensor(immagine), 
                'annotazione': torch.from_numpy(annotazione) }


class MyTransforms(object):
    """Crop centrale di dimension data .

    Args:
        output_size (tuple or int): Dmensione input desiderata.
    """

    def __init__(self, output_size, phase):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.phase = phase

    def __call__(self, esempio):
        immagine, annotazione = esempio['immagine'], esempio['annotazione']

        if ( self.phase == 'train'):
    
            RRC = transforms.RandomResizedCrop(self.output_size)
            RHF = transforms.RandomHorizontalFlip()
            immagine = RRC(immagine)
            immagine = RHF(immagine)
        else: 
            RES = transforms.Resize(256)
            CC = transforms.CenterCrop(self.output_size)
            immagine = RES (immagine)
            immagine = CC (immagine)

        return {'immagine': immagine,
                'annotazione': annotazione}