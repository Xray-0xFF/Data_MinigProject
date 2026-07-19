# 🩺 IA appliquée à la santé — Classification de 8 pathologies à partir de 5 symptômes

Mini-projet réalisé dans le cadre du module **Data Mining** — Master Intelligence Artificielle et Sciences des Données (IASD), Université Mohammed Premier, Faculté des Sciences, Oujda.

**Année universitaire :** 2025 – 2026
**Encadré par :** Mr. Aissa KERKOUR El MIAD
**Réalisé par :**
- KAZDABI NBARK
- JAIL HAMID

---

## 📌 Présentation du projet

Ce projet explore le diagnostic médical assisté par intelligence artificielle à partir de **cinq symptômes cliniques** (fièvre, maux de tête, toux, fatigue, douleurs corporelles), afin de prédire laquelle de **huit pathologies courantes** un patient est susceptible d'avoir :

`Rhume` · `Paludisme` · `Toux` · `Asthme` · `Fièvre normale` · `Courbatures` · `Écoulement nasal` · `Dengue`

Le dataset utilisé est un jeu de données **synthétique de 5 000 patients**, généré avec NumPy/Pandas selon des corrélations médicalement réalistes, parfaitement équilibré (625 patients par pathologie).

## 🎯 Objectifs

- Explorer et prétraiter des données cliniques (analyse univariée/bivariée, outliers, normalisation)
- Comparer plusieurs modèles de classification supervisée
- Appliquer des techniques de clustering non supervisé pour valider la structure des données
- Extraire des règles de diagnostic interprétables via un arbre de décision
- Comparer une approche low-code (Orange) à une approche codée (Python)

## 🛠️ Outils et méthodes

| Volet | Outils |
|---|---|
| Exploration & workflow visuel | **Orange Data Mining** (Box Plot, Scatter Plot, Mosaic Display, Rank, PCA, Heat Map) |
| Prétraitement | Outliers (LOF), Impute, Normalisation (Z-score) |
| Modèles supervisés | k-NN, Naive Bayes, Random Forest, Réseau de neurones, SVM, Régression Logistique |
| Clustering non supervisé | K-means, Classification Ascendante Hiérarchique (CAH) |
| Extraction de connaissances | Arbre de décision (règles de diagnostic) |
| Approfondissement Python | scikit-learn, SMOTE, réseaux bayésiens, règles d'association (Apriori) |

## 📊 Résultats clés

- **Random Forest** : modèle le plus performant → **Accuracy 94,4 %**, **AUC 0,978**
- **k-NN** : excellente alternative, très rapide (0,137s d'entraînement), AUC 0,960
- Les pathologies **Body Ache, Dengue et Malaria** sont classifiées avec une précision quasi parfaite (>98%)
- **Common Cold** est la pathologie la plus difficile à distinguer (chevauchement de symptômes avec Asthma)
- L'arbre de décision extrait **16 règles de diagnostic interprétables**, dont plusieurs à 100% de fiabilité
- Le clustering K-means et la CAH confirment une structure naturelle des données en 8 groupes cohérents avec les pathologies réelles



## 📄 Rapport complet

Le rapport détaillé (méthodologie, figures, analyses) est disponible 
## 📜 Licence

Projet académique — Master IASD, Université Mohammed Premier.
