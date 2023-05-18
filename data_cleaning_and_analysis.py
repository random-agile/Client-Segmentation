#!/usr/bin/env python
# coding: utf-8

# # OC PROJET 4 - CLIENT SEGMENTATION
# #### CLEANING AND ANALYSIS NOTEBOOK
# <br></br>
# ### SOMMAIRE
# - <a href="#C1">I. Nettoyage et fusion des données</a>
#     - 1. Importation des librairies
#     - 2. Paramétrages Data Visualisation
#     - 3. Merging vers un dataframe principal
#     - 4. Gestion des valeurs manquantes et doublons
#     - 5. Conversion des variables
#     - 6. Gestion des outliers
#     - 7. Vérifications et sauvegarde du dataset cleané
#     
# - <a href="#C2">II. Feature Engineering</a>
#      - 1. Création de nouvelles variables
#      - 2. Création de variables de groupes et d'interaction
#      - 3. Création de variables de fréquence d'achat
#      - 4. Création de variables de valeur client  
#     
# - <a href="#C3">III. Exploration des données</a>
#     - 1. Matrice des corrélations
#     - 2. Analyse temporelle
#     - 3. Analyse Quanti/Quanti
#     - 4. Analyse Quanti/Quali
#     - 5. Analyse Quali/Quali
#     - 6. ACP

# # <a name="C1">I. Nettoyage et fusion des données</a>

# <font size="5">1. Importation des librairies</font>

# In[5]:


# importation des librairies
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import font_manager as rcParams
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from scipy.stats import pearsonr
from scipy.stats import f_oneway
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# <font size="5">2. Paramétrages Data Visualisation</font>

# In[72]:


# Ajouter une ombre à la police
shadow = path_effects.withSimplePatchShadow(offset=(1, - 0.75), 
shadow_rgbFace='darkblue', alpha = 0.25)

# changer la police dans les graphiques, les couleurs 
# et augmenter la résolution d'affichage
plt.rcParams['font.family'] = 'Ebrima'
plt.rcParams['text.color'] = 'white'
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.style.use('dark_background')

# set le theme seaborn
sns.set_style('darkgrid', {'axes.facecolor': '0.2',
'text.color': 'white', 'figure.figsize': (20, 16)})
plt.rcParams['figure.facecolor'] = '0.2'

# suppression de l'affichage max des colonnes
pd.set_option('display.max_columns', None)


# <font size="5">3. Merging vers un dataframe principal</font>

# <i>Remarque : j'ai décider de ne pas inclure le dataset geolocation dans le merge étant donné 
# que nous avons déjà les infos similaires dans les autres dataset (code postal, ville, état)</i>

# In[44]:


# chargement multiple des dataframes
df_customers = pd.read_csv('olist_customers_dataset.csv')
df_orders = pd.read_csv('olist_orders_dataset.csv')
df_order_review = pd.read_csv('olist_order_reviews_dataset.csv')
df_order_items = pd.read_csv('olist_order_items_dataset.csv')
df_order_payments = pd.read_csv('olist_order_payments_dataset.csv')
df_products = pd.read_csv('olist_products_dataset.csv')
df_products_cat = pd.read_csv('product_category_name_translation.csv')
df_sellers = pd.read_csv('olist_sellers_dataset.csv')


# In[45]:


# fusion des dataframes vers un dataframe principal
df_merged = pd.merge(df_customers, df_orders, on='customer_id', how = 'left')
df_merged_a = pd.merge(df_merged, df_order_review, how = 'left', on = 'order_id')
df_merged_b = pd.merge(df_merged_a, df_order_items, how = 'left', on = 'order_id')
df_merged_c = pd.merge(df_merged_b, df_products, how = 'left', on = 'product_id')
df_merged_d = pd.merge(df_merged_c, df_products_cat, how = 'left', on = 'product_category_name')
df_merged_e = pd.merge(df_merged_d, df_sellers, how = 'left', on = 'seller_id')
df = pd.merge(df_merged_e, df_order_payments, how = 'left', on = 'order_id')


# Faire un diagramme sur le dataframe, là ou on merge, sur quelle clé etc... (voir exemple sur kaggle) diagram database site
# 
# Suite : je n'ai pas trouver d'exemple

# In[46]:


df.shape


# In[47]:


# affichage du dataframe principal
df


# <font size="5">4. Gestion des doublons et valeurs manquantes</font>

# In[48]:


df.isnull().sum()


# Note : "La segmentation RFM prend en compte la Récence (date de la dernière commande), la Fréquence des commandes et le Montant (de la dernière commande ou sur une période donnée) pour établir des segments de clients homogènes."
# 
# Source : https://www.definitions-marketing.com/definition/segmentation-rfm/

# In[49]:


# suppression des variables inutiles
df = df.drop(columns = ['order_approved_at', 'order_delivered_carrier_date',
'order_estimated_delivery_date', 'review_id', 'review_answer_timestamp',
'order_item_id', 'payment_sequential', 'product_name_lenght', 
'product_description_lenght', 'product_photos_qty', 'product_weight_g',
'product_length_cm', 'product_height_cm', 'product_width_cm',
'product_category_name', 'shipping_limit_date', 'customer_zip_code_prefix',
'seller_zip_code_prefix', 'review_creation_date'])


# In[50]:


# suppression des variables qui contiennent trop de valeurs manquantes
# et qui ne peuvent pas être restituées ou recalculées
df = df.drop(columns = ['review_comment_title', 'review_comment_message'])


# In[51]:


# suppression des valeurs manquantes
df = df.dropna()


# In[52]:


# check des doublons
df.loc[df.duplicated(keep = False),:]


# In[53]:


# Suppression des doublons
df.drop_duplicates(inplace=True)


# In[54]:


df


# <font size="5">5. Conversion des variables</font>

# In[55]:


df.dtypes


# In[56]:


# /!\ le saut à la ligne pour respecter la convention
# PEP 8 casse le code, j'ai donc laisser intentionnelement
# des lignes de plus de 79 charactères dans cette cellule

# création de deux nouvelles variables en faisant
# une conversion vers le format datetime
df['order_purchase_datetime'] = pd.to_datetime(df['order_purchase_timestamp'])
df['order_delivered_datetime'] = pd.to_datetime(df['order_delivered_customer_date'])

# supression des anciennes colonnes du mauvais dtype
df = df.drop(columns=['order_purchase_timestamp', 'order_delivered_customer_date'])


# In[57]:


df.dtypes


# In[43]:


df


# In[19]:


# fonction pour checker si les variables 
# ne contiennent pas de valeurs infinies
def infinite_check(df):
    for col in df.columns:
        if df[col].dtype.kind in 'biufc' and np.isinf(df[col]).any():
            print(f"La colonne {col} contient des valeurs égales à infini.")


# In[20]:


infinite_check(df)


# <font size="5">6. Gestion des outliers</font>

# In[21]:


df.describe()


# Analyse du describe : 
# - Pas de valeurs négatives à signaler
# - Pour le code postal : la valeur max est 99980 donc visiblement pas de outliers ici
# - Le review score va de 1 à 5 donc pas d'anomalie ici
# - Vérifier pour le 6735€ prix max et 409€ de frais de port max
# - Il existe un paiement à 13664€ alors que le prix le plus haut d'un produit est 6735€
# - Attention également aux paiments avec une valeur égale à 0

# In[58]:


# quelques corrections après l'analyse du describe
df = df[df['payment_value'] <= 6735]
df = df[df['payment_value'] > 0]


# In[73]:


plt.rcParams['font.family'] = 'Ebrima'
# on vérifie avec les boxplot
list_columns = ['review_score', 'price', 'freight_value', 
'payment_installments', 'payment_value']

for columns_name in list_columns:
    plt.grid(False)
    plt.title(f"Histogramme {columns_name}")
    plt.boxplot(df[columns_name], vert = False)
    plt.xticks(color = 'white')
    plt.yticks(color = 'white')
    plt.show()
    print(columns_name)


# In[68]:


# correction après l'analyse des boxplot
df = df[df['price'] <= 5000]
df = df[df['payment_value'] <= 5000]
df = df[df['freight_value'] <= 360]
df.shape


# In[74]:


# on vérifie que la distribution des variables est correcte
list_columns = ['review_score', 'price', 'freight_value', 
'payment_installments', 'payment_value']

for columns_name in list_columns:
    plt.title(f"Histogramme {columns_name}")
    plt.hist(df[columns_name])
    plt.xticks(color = 'white')
    plt.yticks(color = 'white')
    plt.grid(False)
    plt.show()
    print(columns_name)


# In[80]:


# Tracer la distribution des prix
sns.kdeplot(df['price'], shade = True)
plt.xlabel('Prix', color = 'white')
plt.gca().yaxis.set_label_coords(-0.175, 0.5)
plt.ylabel('Densité', color = 'white', rotation = 360)
plt.title('Distribution des prix des produits', color = 'white')
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.xlim(0, 1000)
plt.show()


# <font size="5">7. Vérification et sauvegarde du dataframe cleané</font>

# In[81]:


df


# In[82]:


# vérification des types
df.dtypes


# In[83]:


# vérification des doublons
df.loc[df.duplicated(keep = False),:]


# In[84]:


# vérification des valeurs nulles
df.isnull().sum()


# In[85]:


# fonction pour checker si les variables ne contiennent pas de valeurs infinies
def infinite_check(df):
    for col in df.columns:
        if df[col].dtype.kind in 'biufc' and np.isinf(df[col]).any():
            print(f"La colonne {col} contient des valeurs égales à infini.")


# In[88]:


# vérification des valeurs infinies
infinite_check(df)


# In[87]:


# sauvegarde du dataframe
df.to_csv('olist_master.csv', index = False)


# # <a name="C2">II. Feature Engineering</a>

# <font size="5">1. Ajout de variables temporelles</font>

# In[89]:


# création des variables temporelles
df['month'] = pd.to_datetime(df['order_purchase_datetime']).dt.month
df['year'] = pd.to_datetime(df['order_purchase_datetime']).dt.year


# In[90]:


# Créer la variable temps de livraison
df['delivery_time_days'] = (pd.to_datetime(df['order_delivered_datetime']) - 
pd.to_datetime(df['order_purchase_datetime'])).dt.days


# <font size="5">2. Création de variables de groupes et d'interaction</font>

# In[91]:


# création de la variable prix moyen par catégorie de produit
average_price_cat = df.groupby('product_category_name_english')['price'].mean()
df['average_price_cat'] = df['product_category_name_english'].map(average_price_cat)


# <font size="5">3. Création de variables de fréquence d'achat</font>

# In[97]:


# /!\ le saut à la ligne pour respecter la convention
# PEP 8 casse le code, j'ai donc laisser intentionnelement
# des lignes de plus de 79 charactères dans cette cellule

# Créer la variable du nombre d'achats sur les 3 derniers mois
limit_date = pd.to_datetime('2018-09-01')
df['nb_purchase_last_3_months'] = df[df['order_purchase_datetime'] > 
(limit_date - pd.DateOffset(months=3))].groupby('customer_id')['customer_id'].transform('count')
 # remplacer les valeurs manquantes par 0
df['nb_purchase_last_3_months'].fillna(0, inplace=True)


# <font size="5">4. Création de variables de valeur client</font>

# In[102]:


# /!\ le saut à la ligne pour respecter la convention
# PEP 8 casse le code, j'ai donc laisser intentionnelement
# des lignes de plus de 79 charactères dans cette cellule

# Créer la variable du montant total des achats effectués par le client
total_purchases = df.groupby('customer_id')['payment_value'].sum()
df['total_amount_client'] = df['customer_id'].map(total_purchases)

# Créer la variable du nombre de produits différents achetés par le client
nb_different_product = df.groupby('customer_id')['product_category_name_english'].nunique()
df['nb_different_product_client'] = df['customer_id'].map(nb_different_product)

# Créer la variable du nombre de jours depuis le dernier achat
last_purchase = df.groupby('customer_id')['order_purchase_datetime'].max()
df['days_since_last_purchase'] = (limit_date - df['customer_id'].map(last_purchase)).dt.days
# remplacer les valeurs manquantes avec le temps écoulé depuis la dernière observation
df['days_since_last_purchase'].fillna((limit_date - df['order_purchase_datetime']).dt.days, inplace=True)


# In[103]:


df


# In[104]:


df.describe()


# In[106]:


# sauvegarde du dataframe
df.to_csv('olist_master.csv', index=False)


# In[107]:


df.dtypes


# In[131]:


df = pd.read_csv('olist_master.csv')

# on convertit à nouveau les données en format datetime car
# le type à changer entre les chargement de df
df['order_purchase_datetime'] = pd.to_datetime(df['order_purchase_datetime'])
df['order_delivered_datetime'] = pd.to_datetime(df['order_delivered_datetime'])


# # <a name="C3">III. Exploration des données</a>

# <font size="5">1. Matrice des corrélations</font>

# In[132]:


sns.set(font_scale = 0.7, rc = {'axes.labelcolor': 'white', 
'xtick.color': 'white', 'ytick.color': 'white'})
plt.title("Matrice des correlations clients", color = 'white')

sns.heatmap(df.corr(), center = 1, annot = True, 
fmt = ".2f", vmin =- 1, vmax = 1)

# Ajouter une bordure personnalisée avec une couleur de fond grise
fig = plt.gcf()
fig.set_size_inches(8, 6)
fig.patch.set_facecolor('0.2')

plt.show()


# Analyse de la matrice des correlations :
# - Très forte corrélation positive entre le prix et valeur de paiment (0,89)
# - Corrélation positive entre le prix et le nombre de paiment (0,30)
# - Corrélation positive entre le prix et les frais de port (0,41)
# - Corrélation négative entre le temps de livraison et le review_score (-0.32)

# <font size="5">2. Analyse Temporelle</font>

# In[134]:


# /!\ le saut à la ligne pour respecter la convention
# PEP 8 casse le code, j'ai donc laisser intentionnelement
# des lignes de plus de 79 charactères dans cette cellule

# set du style sns après la matrice
sns.set_style('darkgrid', {'axes.facecolor': '0.2', 
'text.color': 'white', 'figure.figsize': (20, 16)})
plt.rcParams['figure.facecolor'] = '0.2'
# Créer une colonne de date à partir 
# de la colonne order_purchase_datetime
df['order_purchase_date'] = pd.to_datetime(df['order_purchase_datetime']).dt.date

# Agréger les données par mois pour le nombre d'achats par client
monthly_purchases = df.groupby(['order_purchase_date'])['customer_id'].nunique()

# Tracer le graphique en ligne
plt.plot(monthly_purchases.index, monthly_purchases.values)
plt.xlabel('Date', color = 'white', fontsize = 10)
plt.gca().yaxis.set_label_coords(-0.15, 0.5)
plt.ylabel('Achats', color = 'white', rotation = '360', fontsize = 10)
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.title("Nombre d'achats clients par mois", fontsize = 12)
plt.grid(False)
plt.show()


# In[112]:


mpl.rcParams['text.color'] = 'white'
# Extraire le mois et l'année à partir 
# de la colonne 'order_purchase_timestamp'
df['month_year'] = df['order_purchase_datetime'].dt.to_period('M')

# Compter le nombre de commandes par mois
orders_per_month = df['month_year'].value_counts().sort_index()

# Afficher l'évolution du nombre de commandes par mois
plt.figure(figsize = (10, 6))
orders_per_month.plot(kind = 'line', marker = 'o')
plt.title("Évolution du nombre de commandes par mois", size = 14)
plt.gca().yaxis.set_label_coords(-0.225, 0.5)
plt.ylabel("Nombre de commandes", color = 'white', rotation = 360, size = 12)

plt.xticks(color = 'white', size = 10)
plt.yticks(color = 'white', size = 10)
plt.grid(False)
plt.show()


# In[115]:


sns.barplot(x = "review_score", y = "delivery_time_days", 
data = df, color='royalblue')
plt.xlabel('Review Score', size = 10, color = 'white')
plt.gca().yaxis.set_label_coords( - 0.28, 0.5)
plt.ylabel('Temps de livraison (jours)', 
size = 10, rotation = 360, color = 'white')
plt.title('Review Score en fonction du temps de livraison')
plt.xticks(fontsize = 10, color = 'white')
plt.yticks(fontsize = 10, color = 'white')
plt.grid(False)
plt.show()


# <font size="5">3. Analyse Quanti/Quanti</font>

# In[136]:


# on stocke la valeur du coefficient de pearson dans une variable
r_value = st.pearsonr(df["price"], df["freight_value"])[0]
# même chose pour la covariance
cov_value = np.cov(df["price"], df["freight_value"], ddof = 0)[1,0]

#customisation du graph
plt.title("Prix en fonction des frais de port", fontsize = 14)
plt.xticks(fontsize = 10, color = 'white')
plt.yticks(fontsize = 10, color = 'white')

ax = sns.regplot(x = "price", y = "freight_value", data = df, 
color = 'springgreen', line_kws = {"color":"royalblue"}, 
scatter_kws = {"alpha":0.2, "edgecolor":"springgreen"}, marker = 'o')

plt.xlabel("Prix", color = 'white', size = 12, labelpad = 10)
plt.gca().yaxis.set_label_coords( - 0.233, 0.5)
plt.ylabel("Frais de port", color = 'white', size = 12, rotation = 360)
plt.text( - 2000,  - 80, 'r = {}'.format(r_value), color = 'white', size = 8)
plt.text( - 2000,  - 100.5, 'Cov(x,y) = {}'.format(cov_value), 
color = 'white', size = 8)
ax.grid(False)
# Appliquer les effets de texte aux titres
for title in [ax.title, ax.xaxis.label, ax.yaxis.label]:
    title.set_path_effects([shadow])

# Appliquer les effets de texte aux ticks
for tick in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
    tick.set_path_effects([shadow])
plt.show()


# <font size="5">4. Analyse Quanti/Quali</font>

# In[118]:


category_counts = df['product_category_name_english'].value_counts().head(10)

plt.bar(category_counts.index, category_counts.values)
plt.ylabel('Count', color = 'white', rotation = 360)
plt.title('Top 10 Product Categories', color = 'white')
plt.gca().yaxis.set_label_coords( - 0.15, 0.5)
plt.xticks(rotation = 45, color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.show()


# In[120]:


category_counts = df['customer_city'].value_counts().head(10)

plt.bar(category_counts.index, category_counts.values)
plt.ylabel('Count', color = 'white', rotation = 360)
plt.title('Top 10 Customers City', color = 'white')
plt.gca().yaxis.set_label_coords( - 0.15, 0.5)
plt.xticks(rotation = 45, color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.show()


# In[119]:


df.dtypes


# <font size="5">5. Analyse Quali/Quali</font>

# In[121]:


# Définissez le dictionnaire de correspondance
correspondance = {1: 'E', 2: 'D', 3: 'C', 4: 'B', 5: 'A'}

# Utilisez la méthode map() pour créer 
# une nouvelle colonne avec les lettres correspondantes
df['review_letter'] = df['review_score'].map(correspondance)


# In[122]:


# Créer une table de contingence entre les variables qualitatives
contingency_table = pd.crosstab(df['review_letter'], df['payment_type'])

# Afficher un graphique en barres empilées
plt.figure(figsize = (10, 6))
sns.heatmap(contingency_table, annot = True, fmt = "d", cmap = "Blues")
plt.title("Table de contingence entre Review du client et type de paiement")
plt.xlabel("Type de paiement", color = "white")
plt.ylabel("Review du client", color = "white")
plt.xticks(color = "white")
plt.yticks(color = "white")

plt.show()


# <font size="5">6. ACP</font>

# In[123]:


df = pd.read_csv('olist_master.csv')


# In[126]:


df.shape


# In[137]:


# suppression des variables non numériques
df_pca = df.select_dtypes(include = [np.number])
df_pca = df_pca.drop(columns = ['month', 'year',
'nb_purchase_last_3_months', 'payment_value', 'average_price_cat'])
X = df_pca.values
names = df_pca.index
features = df_pca.columns

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# préciser le nombre de components
n_components = 7

# Créer un objet PCA avec le nombre de composantes souhaité
pca = PCA(n_components = n_components)
pca.fit(X_scaled)

pca.explained_variance_ratio_
scree = (pca.explained_variance_ratio_*100).round(2)
scree

# on arrondit à la somme cumulée
scree_cum = scree.cumsum().round()
scree_cum

x_list = range(1, n_components+1)
list(x_list)

plt.bar(x_list, scree)
plt.plot(x_list, scree_cum, c = "royalblue",marker = 'o')
plt.xlabel("rang de l'axe d'inertie", color = 'white')
plt.gca().yaxis.set_label_coords(-0.2, 0.5)
plt.ylabel("pourcentage d'inertie", color = 'white', rotation = 360)
plt.title("Eboulis des valeurs propres", color = 'white')
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.grid(False)
plt.show(block = False)
x,y = 0,1
fig, ax = plt.subplots(figsize = (10, 9))
for i in range(0, pca.components_.shape[1]):
    ax.arrow(0,
             0,  # Start the arrow at the origin
             pca.components_[0, i],  #0 for PC1
             pca.components_[1, i],  #1 for PC2
             head_width = 0.07,
             head_length = 0.07, 
             width = 0.02,              )

    plt.text(pca.components_[0, i] + 0.05,
             pca.components_[1, i] + 0.05,
             features[i], fontsize = 8)
    
# affichage des lignes horizontales et verticales
plt.plot([-1, 1], [0, 0], color = 'grey', ls = '--')
plt.plot([0, 0], [-1, 1], color = 'grey', ls = '--')


# nom des axes, avec le pourcentage d'inertie expliqué
plt.xlabel('F{} ({}%)'.format(x + 1, 
round(100 * pca.explained_variance_ratio_[x], 1)), color = 'white')
plt.ylabel('F{} ({}%)'.format(y + 1, round(
100 * pca.explained_variance_ratio_[y],1)), color = 'white', rotation = 360)

plt.title("Cercle des corrélations (F{} et F{})".format(x + 1, y + 1))
plt.xticks(color = 'white')
plt.yticks(color = 'white')
plt.gca().yaxis.set_label_coords( - 0.1, 0.5)
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
plt.axis('equal')
plt.grid(False)
plt.show(block = False)

