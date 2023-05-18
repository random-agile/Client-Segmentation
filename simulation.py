#!/usr/bin/env python
# coding: utf-8

# # OC PROJET 4 - CLIENT SEGMENTATION
# #### SIMULATION NOTEBOOK
# <br></br>
# ### SOMMAIRE
# - <a href="#C1">I. Simulation</a>
#     - 1. 
#     - 2. 
#     - 3.
#     
# - <a href="#C2">II. Proposition de Maintenance</a>
#      - 1. 
#      - 2.
#      - 3. 
#      - 4.

# In[70]:


import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as fm, rcParams
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# <font size="5">Paramétrages Data Visualisation</font>

# In[65]:


# Ajouter une ombre à la police
shadow = path_effects.withSimplePatchShadow(offset=(1,-0.75), shadow_rgbFace='darkblue', alpha=0.25)

# Utiliser la police dans les graphiques, changer la couleur et augmenter la résolution d'affichage
plt.rcParams['font.family'] = 'Ebrima'
plt.rcParams['text.color'] = 'white'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.style.use('dark_background')
# set le theme seaborn
sns.set_style('darkgrid', {'axes.facecolor': '0.2','text.color': 'white','figure.figsize': (20, 16)})
plt.rcParams['figure.facecolor'] = '0.2'
# définition des paramètres kwargs typo
text_kwargs = dict(ha='center', va='center', fontsize=10, color='white')

# suppression de l'affichage max des colonnes
pd.set_option('display.max_columns', None)


# In[66]:


# chargement du df master et conversion des variables temporelles au format datetime

df = pd.read_csv('olist_master.csv')

df['order_purchase_datetime'] = pd.to_datetime(df['order_purchase_datetime'])
df['order_delivered_datetime'] = pd.to_datetime(df['order_delivered_datetime'])
df['review_creation_datetime'] = pd.to_datetime(df['review_creation_datetime'])


# # TEST DE K MEANS SUR 3 PERIODES DE 6 MOIS

# In[4]:


df['order_purchase_datetime'].max()


# In[5]:


df['order_purchase_datetime'].min()


# In[17]:


# Convertir la colonne 'order_purchase_datetime' en format de date
df['order_purchase_datetime'] = pd.to_datetime(df['order_purchase_datetime'])

# Filtrer les échantillons pour chaque critère
df_1st_sem_2017 = df[(df['order_purchase_datetime'].dt.year == 2017) & (df['order_purchase_datetime'].dt.month <= 6)]
df_2nd_sem_2017 = df[(df['order_purchase_datetime'].dt.year == 2017) & (df['order_purchase_datetime'].dt.month > 6)]
df_1st_sem_2018 = df[(df['order_purchase_datetime'].dt.year == 2018) & (df['order_purchase_datetime'].dt.month <= 6)]


# In[7]:


df_1st_sem_2017.shape


# In[14]:


df_2nd_sem_2017.shape


# In[15]:


df_1st_sem_2018.shape


# # SIMULATION SEMESTRE 1 2017

# In[21]:


df_1st_sem_2017.describe()


# In[48]:


# Chargement des données
data = df_1st_sem_2017

# Sélection des colonnes pertinentes pour la segmentation
selected_columns = ['total_amount_client', 'review_score', 'days_since_last_purchase', 'payment_installments', 'price']
X = data[selected_columns]

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Ajout des étiquettes de clusters au dataframe d'origine
data['cluster'] = kmeans.labels_

# Analyse des clusters
cluster_counts = data['cluster'].value_counts()
print("Nombre de clients par cluster:")
print(cluster_counts)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['total_amount_client'], X['days_since_last_purchase'], 
c=kmeans.labels_, cmap='viridis', alpha=0.5, linewidths=0.5)
plt.xlabel('Argent total dépensé par le client', color = 'white')
plt.gca().yaxis.set_label_coords(-0.2, 0.5)
plt.ylabel('Dernier Achat (jours)', color = 'white', rotation = 360)
plt.title('Segmentation des clients premier semestre 2017')
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.colorbar()
plt.show()


# In[49]:


data_cluster = data[selected_columns]
data_cluster['cluster'] = kmeans.labels_
# Calculer les moyennes des variables pour chaque cluster
cluster_means = data_cluster.groupby('cluster').mean()

# Afficher les moyennes des variables pour chaque cluster
print(cluster_means)


# In[50]:


# Calculer les moyennes des variables pour chaque cluster
cluster_describe = data_cluster.groupby('cluster').describe()

# Définir la largeur maximale des colonnes de sortie
pd.set_option('display.width', 110)
# Afficher les moyennes des variables pour chaque cluster
print(cluster_describe)


# # SIMULATION SEMESTRE 2 2017

# In[69]:


# Chargement des données
data = df_2nd_sem_2017

# Sélection des colonnes pertinentes pour la segmentation
selected_columns = ['total_amount_client', 'review_score', 'days_since_last_purchase', 'payment_installments', 'price']
X = data[selected_columns]

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Ajout des étiquettes de clusters au dataframe d'origine
data['cluster'] = kmeans.labels_

# Analyse des clusters
cluster_counts = data['cluster'].value_counts()
print("Nombre de clients par cluster:")
print(cluster_counts)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['total_amount_client'], X['days_since_last_purchase'], c=kmeans.labels_
, cmap='viridis', alpha=0.5, linewidths=0.5)
plt.xlabel('Argent total dépensé par le client', color = 'white')
plt.gca().yaxis.set_label_coords(-0.2, 0.5)
plt.ylabel('Dernier Achat (jours)', color = 'white', rotation = 360)
plt.title('Segmentation des clients deuxième semestre 2017')
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.colorbar()
plt.show()


# In[52]:


data_cluster = data[selected_columns]
data_cluster['cluster'] = kmeans.labels_
# Calculer les moyennes des variables pour chaque cluster
cluster_means = data_cluster.groupby('cluster').mean()

# Afficher les moyennes des variables pour chaque cluster
print(cluster_means)


# In[53]:


# Calculer les moyennes des variables pour chaque cluster
cluster_describe = data_cluster.groupby('cluster').describe()

# Définir la largeur maximale des colonnes de sortie
pd.set_option('display.width', 110)
# Afficher les moyennes des variables pour chaque cluster
print(cluster_describe)


# # SIMULATION SEMESTRE 1 2018

# In[54]:


# Chargement des données
data = df_1st_sem_2018

# Sélection des colonnes pertinentes pour la segmentation
selected_columns = ['total_amount_client', 'review_score', 'days_since_last_purchase', 'payment_installments', 'price']
X = data[selected_columns]

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Ajout des étiquettes de clusters au dataframe d'origine
data['cluster'] = kmeans.labels_

# Analyse des clusters
cluster_counts = data['cluster'].value_counts()
print("Nombre de clients par cluster:")
print(cluster_counts)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['total_amount_client'], X['days_since_last_purchase'], c=kmeans.labels_
, cmap='viridis', alpha=0.5, linewidths=0.5)
plt.xlabel('Argent total dépensé par le client', color = 'white')
plt.gca().yaxis.set_label_coords(-0.2, 0.5)
plt.ylabel('Dernier Achat (jours)', color = 'white', rotation = 360)
plt.title('Segmentation des clients premier semestre 2018')
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.colorbar()
plt.show()


# In[55]:


data_cluster = data[selected_columns]
data_cluster['cluster'] = kmeans.labels_
# Calculer les moyennes des variables pour chaque cluster
cluster_means = data_cluster.groupby('cluster').mean()

# Afficher les moyennes des variables pour chaque cluster
print(cluster_means)


# In[57]:


# Calculer les moyennes des variables pour chaque cluster
cluster_describe = data_cluster.groupby('cluster').describe()

# Définir la largeur maximale des colonnes de sortie
pd.set_option('display.width', 110)
# Afficher les moyennes des variables pour chaque cluster
print(cluster_describe)


# # TEST DE K MEANS SUR 2 PERIODES DE 1 AN

# In[34]:


# Convertir la colonne 'order_purchase_datetime' en format de date
df['order_purchase_datetime'] = pd.to_datetime(df['order_purchase_datetime'])

# Filtrer les échantillons pour chaque critère
df_2017 = df[(df['order_purchase_datetime'].dt.year == 2017)]
df_2018 = df[(df['order_purchase_datetime'].dt.year == 2018)]


# In[11]:


df_2017.shape


# In[12]:


df_2018.shape


# # SIMULATION POUR 2017

# In[58]:


# Chargement des données
data = df_2017

# Sélection des colonnes pertinentes pour la segmentation
selected_columns = ['total_amount_client', 'review_score', 'days_since_last_purchase', 'payment_installments', 'price']
X = data[selected_columns]

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Ajout des étiquettes de clusters au dataframe d'origine
data['cluster'] = kmeans.labels_

# Analyse des clusters
cluster_counts = data['cluster'].value_counts()
print("Nombre de clients par cluster:")
print(cluster_counts)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['total_amount_client'], X['days_since_last_purchase'], c=kmeans.labels_,
cmap='viridis', alpha=0.5, linewidths=0.5)
plt.xlabel('total_amount_client', color = 'white')
plt.ylabel('days_since_last_purchase', color = 'white')
plt.title('Segmentation des clients pour 2017')
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.colorbar()
plt.show()


# In[59]:


data_cluster = data[selected_columns]
data_cluster['cluster'] = kmeans.labels_
# Calculer les moyennes des variables pour chaque cluster
cluster_means = data_cluster.groupby('cluster').mean()

# Afficher les moyennes des variables pour chaque cluster
print(cluster_means)


# In[60]:


# Calculer les moyennes des variables pour chaque cluster
cluster_describe = data_cluster.groupby('cluster').describe()

# Définir la largeur maximale des colonnes de sortie
pd.set_option('display.width', 110)
# Afficher les moyennes des variables pour chaque cluster
print(cluster_describe)


# # SIMULATION POUR 2018

# In[61]:


# Chargement des données
data = df_2018

# Sélection des colonnes pertinentes pour la segmentation
selected_columns = ['total_amount_client', 'review_score', 'days_since_last_purchase', 'payment_installments', 'price']
X = data[selected_columns]

# Prétraitement des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'algorithme K-means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X_scaled)

# Ajout des étiquettes de clusters au dataframe d'origine
data['cluster'] = kmeans.labels_

# Analyse des clusters
cluster_counts = data['cluster'].value_counts()
print("Nombre de clients par cluster:")
print(cluster_counts)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
plt.scatter(X['total_amount_client'], X['days_since_last_purchase'], c=kmeans.labels_
, cmap='viridis', alpha=0.5, linewidths=0.5)
plt.xlabel('total_amount_client', color = 'white')
plt.ylabel('days_since_last_purchase', color = 'white')
plt.title('Segmentation des clients pour 2018')
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.colorbar()
plt.show()


# In[62]:


data_cluster = data[selected_columns]
data_cluster['cluster'] = kmeans.labels_
# Calculer les moyennes des variables pour chaque cluster
cluster_means = data_cluster.groupby('cluster').mean()

# Afficher les moyennes des variables pour chaque cluster
print(cluster_means)


# In[63]:


# Calculer les moyennes des variables pour chaque cluster
cluster_describe = data_cluster.groupby('cluster').describe()

# Définir la largeur maximale des colonnes de sortie
pd.set_option('display.width', 110)
# Afficher les moyennes des variables pour chaque cluster
print(cluster_describe)


# In[ ]:




