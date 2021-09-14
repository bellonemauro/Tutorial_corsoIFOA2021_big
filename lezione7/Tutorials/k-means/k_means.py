#  +---------------------------------------------------------------------------+
#  |                                                                           |
#  |  IFOA2021 - BIG DATA e Analisi dei Dati                                   |
#  |  Tutorial : k-means clustering                                            |
#  |                                                                           |
#  |  Autore: Mauro Bellone                                                    |
#  |  Released under BDS License                                               |
#  +---------------------------------------------------------------------------+ 


"""
Questo tutorial è pensato per mostrare il funzionamento dell'algoritmo di clusterizzazione k-means  
"""
print(__doc__)


from numpy import unique
from numpy import where
from matplotlib import pyplot

# affinity propagation clustering
from sklearn.datasets import make_classification
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans

# crea dati random usando un generatore di sklearn, i dati sono composti 
# n_samples = 40 campioni 
# aventi 2 features (quindi 2 dimensioni)
# e 2 possibili annotazioni (quindi 2 centroidi) centers = 2
# random_state ci serve a cambiare i campioni generati come un seed, proviamo valori 50 e 40 
dati, annotazioni = make_blobs(n_samples = 40, n_features=2, centers=2, random_state=40)  
#dati, annotazioni = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Istanziamo il modello 
#modello = AffinityPropagation(preference=-50, random_state=2)
modello = KMeans(n_clusters=2)

# Fittiamo il modello
modello.fit(dati)

# assign a cluster to each example
previsione = modello.predict(dati)

# retrieve unique clusters
clusters = unique(previsione)

# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(previsione == cluster)
	# create scatter of these samples
	pyplot.scatter(dati[row_ix, 0], dati[row_ix, 1])

# show the plot
pyplot.show()