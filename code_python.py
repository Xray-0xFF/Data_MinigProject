
# ÉTAPE 1  IMPORTATION DES BIBLIOTHÈQUES
# =============================================


# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Prétraitement des données
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Modèles de classification supervisée
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Métriques d'évaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Clustering et évaluation non supervisée
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Suppression des avertissements pour une meilleure lisibilité
import warnings
warnings.filterwarnings('ignore')


# ÉTAPE 2 : CHARGEMENT ET EXPLORATION DES DONNÉES (EDA)
# =============================================

# Chargement du jeu de données
df = pd.read_csv('dataset1.csv')

# Affichage des premières lignes pour vérifier le chargement
print("Aperçu des données :")
display(df.head())

# Informations générales sur le DataFrame
print("\nInformations générales :")
df.info()

# Statistiques descriptives des colonnes numériques
print("\nStatistiques descriptives :")
display(df.describe())

# Analyse de la distribution des symptômes (histogrammes)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
numerical_cols = ['Fever', 'Headache', 'Cough', 'Fatigue', 'Body_Pain']

for i, col in enumerate(numerical_cols):
    axes[i].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'Distribution de {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Fréquence')
    axes[i].axvline(df[col].mean(), color='red', linestyle='dashed', linewidth=1, 
                    label=f'Moyenne = {df[col].mean():.2f}')
    axes[i].axvline(df[col].median(), color='green', linestyle='dashed', linewidth=1, 
                    label=f'Médiane = {df[col].median():.2f}')
    axes[i].legend()

fig.delaxes(axes[5])
plt.tight_layout()
plt.show()

# Analyse des corrélations entre symptômes
corr_matrix = df[numerical_cols].corr()
print("Matrice de corrélation :")
display(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matrice de corrélation entre les symptômes')
plt.show()


# ÉTAPE 3 : PRÉPARATION DES DONNÉES (PREPROCESSING)
# =============================================

# Séparation des features (X) et de la target (y)
X = df.drop('Disease', axis=1)
y = df['Disease']

print(f"Dimensions de X : {X.shape}")
print(f"Dimensions de y : {y.shape}")
print(f"Classes de la target : {y.unique()}")
print(f"Nombre de classes : {len(y.unique())}")

# Encodage de la variable cible (transformation des noms de maladies en nombres)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Affichage du mapping des maladies
disease_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nMapping des maladies :")
for disease, code in disease_mapping.items():
    print(f"  {disease} → {code}")

# Division des données en ensembles d'entraînement et de test
# Stratify garantit que la proportion de chaque classe est conservée dans les deux ensembles
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.3, 
    random_state=42,
    stratify=y_encoded
)

print(f"\nTaille de l'ensemble d'entraînement : {X_train.shape[0]} échantillons")
print(f"Taille de l'ensemble de test : {X_test.shape[0]} échantillons")

# Vérification de la distribution des classes après division
print("\nDistribution des classes dans l'entraînement :")
print(pd.Series(y_train).value_counts().sort_index())
print("\nDistribution des classes dans le test :")
print(pd.Series(y_test).value_counts().sort_index())

# Normalisation des données (StandardScaler)
# Ajustement sur l'ensemble d'entraînement uniquement pour éviter le biais de données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStatistiques avant normalisation (entraînement) :")
print(pd.DataFrame(X_train).describe().T[['mean', 'std']])
print("\nStatistiques après normalisation (entraînement) :")
print(pd.DataFrame(X_train_scaled).describe().T[['mean', 'std']])

# Vérification de l'équilibre des classes
class_counts = pd.Series(y_encoded).value_counts().sort_index()
class_names = [le.classes_[i] for i in class_counts.index]

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
bars = plt.bar(class_names, class_counts.values, color='steelblue', edgecolor='black')
plt.title('Distribution des classes')
plt.xlabel('Maladie')
plt.ylabel('Effectif')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 3,
             f'{int(height)}', ha='center', va='bottom', fontsize=10)

plt.subplot(1, 2, 2)
plt.pie(class_counts.values, labels=class_names, autopct='%1.1f%%', 
        colors=sns.color_palette('viridis', len(class_counts)))
plt.title('Répartition en pourcentage des classes')
plt.tight_layout()
plt.show()

# Le jeu de données est parfaitement équilibré : SMOTE n'est pas nécessaire
print(f"\nClasse la moins représentée : {le.classes_[class_counts.idxmin()]} ({class_counts.min()} échantillons)")
print(f"Classe la plus représentée : {le.classes_[class_counts.idxmax()]} ({class_counts.max()} échantillons)")
print(f"Ratio de déséquilibre (max/min) : {class_counts.max()/class_counts.min():.2f}")


# ÉTAPE 4 : MODÉLISATION SUPERVISÉE
# =============================================

# Définition des modèles à évaluer
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM (Linear)': SVC(kernel='linear', random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'MLP Neural Network': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100, 50)),
}

print(f"Nombre de modèles supervisés à évaluer : {len(models)}\n")

# Entraînement et évaluation de chaque modèle
results = {}
for name, model in models.items():
    # Entraînement du modèle
    model.fit(X_train_scaled, y_train)
    
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test_scaled)
    
    # Calcul de la précision
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name:25} - Accuracy: {accuracy:.4f}")


# ÉTAPE 5 : ANALYSE DES MEILLEURS MODÈLES
# =============================================

# Sélection des 3 meilleurs modèles
results_df = pd.DataFrame(results.items(), columns=['Modèle', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False)
print("\nClassement des modèles par performance :")
print(results_df)

# Identification du meilleur modèle
best_model_name = results_df.iloc[0]['Modèle']
best_model = models[best_model_name]
print(f"\nMeilleur modèle : {best_model_name} avec une précision de {results_df.iloc[0]['Accuracy']:.4f}")

# Visualisation des matrices de confusion des 3 meilleurs modèles
best_models = results_df.head(3)['Modèle'].tolist()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, name in enumerate(best_models):
    model = models[name]
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=le.classes_, yticklabels=le.classes_)
    axes[i].set_title(f'Matrice de confusion - {name}')
    axes[i].set_xlabel('Prédit')
    axes[i].set_ylabel('Réel')
    axes[i].tick_params(axis='x', labelrotation=45)

plt.tight_layout()
plt.show()


# ÉTAPE 6 : ANALYSE NON SUPERVISÉE (CLUSTERING)
# =============================================

# K-Means Clustering
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_train_scaled)

# Clustering Hiérarchique
hierarchical = AgglomerativeClustering(n_clusters=8, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_train_scaled)

# Évaluation des clusters
print("\n" + "="*50)
print("ÉVALUATION DU CLUSTERING")
print("="*50)
print(f"K-Means - Silhouette Score: {silhouette_score(X_train_scaled, kmeans_labels):.4f}")
print(f"Hierarchical - Silhouette Score: {silhouette_score(X_train_scaled, hierarchical_labels):.4f}")
print(f"K-Means - Adjusted Rand Index: {adjusted_rand_score(y_train, kmeans_labels):.4f}")
print(f"Hierarchical - Adjusted Rand Index: {adjusted_rand_score(y_train, hierarchical_labels):.4f}")

# Analyse des clusters K-Means
print("\n" + "="*50)
print("ANALYSE DES CLUSTERS K-MEANS")
print("="*50)
print(f"Nombre de clusters : {len(set(kmeans_labels))}")
print(f"Inertie : {kmeans.inertia_:.2f}")


