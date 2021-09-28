#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : Segmentazione                                                 |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 

"""
Questo tutorial è pensato per mostrare il funzionamento di una semplice rete 
neurale addestandola su un task di segmentazione. 
Questo tutorial è rielaborato dai tutorial ufficiali di tensorflow, e ricommentato in italiano.
Tutorial ufficiali disponibili su https://www.tensorflow.org/tutorials
"""
print(__doc__)

# importiamo librerie di tensorflow, se non disponibili semplicemente 
# installabili tramite pip install 
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing 

import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix

# per far vedere le immagini interative 
from IPython.display import clear_output
import matplotlib.pyplot as plt


# funzione di normalizzazione per portare il colore da 0-255 a 0-1
def normalizza(_in_immagine, _in_maschera):
    _in_immagine = tf.cast(_in_immagine, tf.float32) / 255.0
    # Nel dataset le classi segmentate sono indicate con {1, 2, 3}
    # Semplicemente per convenienza computazionale si sottrae 1 
    # le annotazioni diventano quindi {0, 1, 2}.
    _in_maschera -= 1
    return _in_immagine, _in_maschera

# funzione di caricamento immagini e ridimensionamento per adattarsi a u-net
def carica_immagine(datapoint):
    in_immagine = tf.image.resize(datapoint['image'], (128, 128))
    in_maschera = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    in_immagine, in_maschera = normalizza(in_immagine, in_maschera)

    return in_immagine, in_maschera


# classe di data augmentation che eredita da tf.keras.layers.Layer
class dataAugmentation(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        # si opera un randomflip come tecnica di data augmentation, 
        # comunque immagini e maschere (labels) usano lo stesso seme (seed) per 
        # la generazione del numero random, quindi, di fatto, effettuano la stessa  
        # operazione (random) su entrambi i dati 
        self.augment_inputs = preprocessing.RandomFlip(mode="horizontal", seed=seed)
        self.augment_labels = preprocessing.RandomFlip(mode="horizontal", seed=seed)

    # da notare che questo metodo è un metodo ereditato - pure virtual - che viene reimplementato
    def call(self, inputs, labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs, labels

# funzione di visualizzazione 
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Immagine in input', 'Maschera - Ground truth', 'Maschera - Predizione']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# Il modello che usamo è una U-net modificata. 
# La U-net consiste in un encoder (downsampler) e un decoder (upsampler)
# Al fine di imparare un modello migliore e più robusto usiamo una tecnica di transfer learning 
# quindi usando un modello pre-addestrato - MobileNetV2 - come ecoder. 
# Come decoder, useremo un blocco di upsampling già implementato. 
def unet_model(output_channels:int):
    # nel modello di rete U-net si definisce un tensore 
    # in input pari a 128x128 pixel e 3 canali per colore RGB
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Encoder
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Decoder - Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # Ultimo livello del modello
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


# crea la maschera dalla predizione
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

# visualizziamo le predizioni 
def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])    
    else:
        display([sample_image, sample_mask,
                create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# questa callback ci fa vedere come migliora il modello durante le epoche di training
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print ('\nEsempio predetto dopo {} epoche\n'.format(epoch+1))



def add_sample_weights(image, label):
    # I pesi per ogni classe con il vincolo che:
    #     sum(class_weights) == 1.0
    class_weights = tf.constant([2.0, 2.0, 1.0])
    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Crea una immagine di `sample_weights` usando le labels per ogni pixel come un 
    # indice nella `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


    
# entry point
if __name__ == '__main__':
    # scarichiamo il dataset da Oxford pet dataset - https://www.robots.ox.ac.uk/~vgg/data/pets/
    # che è composto da immagini di animali domestici annotati e segmentati
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    # parametri di training
    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

    # in questo caso il dataset contiene già i train/test split quindi semplicemente li carichiamo
    train_images = dataset['train'].map(carica_immagine, num_parallel_calls=tf.data.AUTOTUNE)
    test_images = dataset['test'].map(carica_immagine, num_parallel_calls=tf.data.AUTOTUNE)

    # Costruiamo la pipeline di ingresso, applicando la data augmentation su ogni batch di ingresso 
    train_batches = (
        train_images
        .cache()
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .repeat()
        .map(dataAugmentation())
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    test_batches = test_images.batch(BATCH_SIZE)

    # proviamo una prima visualizzazione delle immagini e la relativa maschera
    for images, masks in train_batches.take(2):
        sample_image, sample_mask = images[0], masks[0]
        display([sample_image, sample_mask])


    # Il modello di base è una rete pre-addestrata - MobileNetV2 - già disponibile in tf.keras.applications. 
    # L'encoder sarà quindi esattamente l'output di questa rete, anche in considerazione dei livelli intermedi di questa rete 
    # NOTA: l'encoder non viene addestrato durante il processo di training.
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Usiamo i seguenti livelli di attivazione
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Creo il modello di estrazione delle features
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    # i parametri dell'encoder non vengono modificati
    down_stack.trainable = False

    # Il modello decoder/upsampler è semplicemente una serie di blocch di upsample che sono già implementati 
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ] 
    # il modello Pix2Pix è una Generative Adversarial Network, o GAN, 
    # e progettato per traduzioni generiche immagine-a-immagine.  
    # Un modelo di discriminazione è addestato per classificare immagini reali (da un dataset) da immagini false (generate), 
    # e il generatore è addestrato per ingannare il discriminatore.


    # istanziamo il modello 
    OUTPUT_CLASSES = 3
    model = unet_model(output_channels=OUTPUT_CLASSES)

    # Dato che questo è un problema di classificazione multiclasse usiamo a CetegoricalCrossentropy 
    # con l'argomento from_logits=True come funzione di loss. 
    # Inoltre usiamo losses.SparseCategoricalCrossentropy(from_logits=True) 
    # dato che le annotazioni sono compste da vettori scalari di interi per ogni classe e per ogni pixel.
    # Quando avviamo la procedura di inferenza, l'annotazione assegnata al pixel corrisponde al canale con il valore più alto.
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # visualizziamo la prima predizione iniziale (pre-training)
    show_predictions()


    EPOCHS = 20
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    # avviamo la procedura di training
    model_history = model.fit(train_batches, epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            validation_data=test_batches,
                            callbacks=[DisplayCallback()])

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    # plottiamo la loss sulla validazione e sul training
    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Loss - Training')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Loss - Validazione')
    plt.title('Loss di training e validazione')
    plt.xlabel('Epoche')
    plt.ylabel('Valore della loss')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()

    # mostriamo qulche predizione sul batch di test
    show_predictions(test_batches, 3)
