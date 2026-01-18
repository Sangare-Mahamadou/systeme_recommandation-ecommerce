# 🧴 Système de Recommandation pour E-commerce Cosmétique

## 📋 Description du Projet

Système de recommandation intelligent pour un site e-commerce de cosmétiques, implémentant trois approches complémentaires :
- **Random Forest** (Baseline) : Modèle interprétable et performant
- **NCF** (Neural Collaborative Filtering) : Deep Learning avec embeddings
- **GRU** : Modélisation séquentielle des sessions utilisateurs

## 🚀 Fonctionnalités

### 🎯 Recommandations Personnalisées
- Recommandations basées sur l'historique utilisateur
- Gestion du cold-start (nouveaux utilisateurs)
- Filtrage par budget, catégorie et marque

### 📊 Dashboard Interactif
- 4 onglets : Vue globale, Recommandations, Analyse produits, Nouveau client
- Visualisations interactives avec Plotly
- KPIs en temps réel et filtres dynamiques

### 🔧 Modèles Avancés
- Trois modèles complémentaires
- Entraînement sur données réelles
- Évaluation comparative des performances

## 🛠 Installation

### Prérequis
- Python 3.8+
- Git

### Installation pas à pas

```bash
# 1. Cloner le dépôt
git clone https://github.com/votre-username/systeme_recommandation-ecommerce.git
cd recommandation-cosmetique

# 2. Créer un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Executions des codes dans l'ordre 
baseline_RF.ipynb ---> embeddings.ipynb ---> séquences.ipynb

# 5. Lancer l'application Streamlit
streamlit run app/streamlit_app.py