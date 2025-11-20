# README Complet – **Nexus Analytics Pro**

_Dashboard analytique avancé – Streamlit + Plotly + IA_

<img src="https://img.shields.io/badge/version-1.0-brightgreen?style=for-the-badge" alt="version"> <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python" alt="python"> <img src="https://img.shields.io/badge/Streamlit-1.30%2B-orange?style=for-the-badge&logo=streamlit" alt="streamlit"> <img src="https://img.shields.io/badge/license-MIT-purple?style=for-the-badge" alt="license">

## Description du projet

**Nexus Analytics Pro** est un tableau de bord analytique moderne, entièrement conçu avec **Streamlit**, offrant :

- Un design premium (glassmorphism + dark mode élégant)
- 6 modules complets : Dashboard, Analyse détaillée, Géographie & Pareto, Simulateur What-If, Machine Learning, Rapports
- Visualisations interactives avec **Plotly**
- Simulateur de scénarios avec élasticité prix intégrée
- Section ML factice prête à accueillir de vrais modèles
- Système de filtres globaux (régions, catégories)
- Téléchargement des données filtrées
- 100 % fonctionnel en mode démo (génère des données réalistes si aucun CSV n’est présent)

## Fonctionnalités principales

| Module                | Fonctionnalités clés                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| Tableau de Bord       | KPIs animés, évolution CA/Profit, top catégories, donut géographique, top vendeurs       |
| Analyse Détaillée     | Histogrammes, boxplots, matrice de corrélation, scatter interactif, heatmap saisonnalité |
| Géographie & Segments | Diagramme de Pareto 80/20, Treemap, Sunburst                                             |
| Simulateur IA         | What-If avec élasticité prix, waterfall d’impact                                         |
| Machine Learning      | Benchmark modèles (fictif), feature importance, statut modèle actif                      |
| Rapports & & Données  | Dataframe interactif, export CSV, logs système                                           |

## Prérequis

```bash
Python >= 3.9
streamlit >= 1.30
pandas
numpy
plotly
joblib
```

## Installation rapide

```bash
# 1. Cloner ou télécharger le projet
git clone https://github.com/tonpseudo/nexus-analytics-pro.git
cd nexus-analytics-pro

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
# ou simplement :
pip install streamlit pandas numpy plotly joblib

# 4. Lancer l'application
streamlit run app.py
```

> Le fichier principal doit s’appeler `app.py` (ou renommez-le et lancez `streamlit run votre_nom.py`).

## Structure des fichiers recommandée

```
nexus-analytics-pro/
│
├── app.py                     ← Le code complet que vous avez partagé
├── requirements.txt           ← (optionnel) dépendances
├── output/
│   └── data/
│       └── cleaned_sales_data.csv   ← Vos vraies données (facultatif)
└── README.md                  ← Ce fichier
```

## Utilisation des données réelles

Le dashboard fonctionne à 100 % sans fichier, grâce à un générateur de données réalistes.

Si vous voulez utiliser vos propres données :

1. Créez le dossier `output/data/`
2. Placez-y votre fichier CSV nommé exactement :  
   `cleaned_sales_data.csv`
3. Le CSV doit contenir au minimum les colonnes suivantes :

```csv
Sale_Date (datetime), Region, Product_Category, Sales_Amount,
Quantity_Sold, Unit_Price, Discount, Profit, Sales_Rep
```

Le dashboard détectera automatiquement le fichier et l’utilisera à la place des données générées.

## Personnalisation rapide

| Élément                    | Où modifier                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| Couleurs principales       | Bloc CSS `:root` (variables `--primary`, `--secondary`, etc.)          |
| Logo & nom                 | Sidebar (emoji + texte)                                                |
| Données générées           | Fonction `generate_dummy_data()`                                       |
| Filtres disponibles        | Sidebar → `df['Region'].unique()` et `df['Product_Category'].unique()` |
| Ajouter de nouvelles pages | Ajouter une entrée dans le `st.radio` + un nouveau `elif`              |

## Captures d’écran (exemples)

_(Vous pouvez les générer vous-même en lançant l’app)_

- Tableau de bord principal → design glassmorphism + cartes animées
- Simulateur IA → waterfall interactif
- Pareto 80/20 + Treemap/Sunburst → vue hiérarchique puissante

## Déploiement (facultatif)

### Streamlit Community Cloud (gratuit)

1. Push le projet sur GitHub
2. Allez sur https://share.streamlit.io
3. Connectez votre repo → `app.py`
4. Déployez !

### Docker (production)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir streamlit pandas numpy plotly joblib

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Licence

**MIT License** – Vous pouvez utiliser, modifier et redistribuer librement ce projet, y compris dans des contextes commerciaux.

## Crédits & Remerciements

- Conçu avec Streamlit & Plotly
- Design inspiré des meilleurs dashboards SaaS 2025
- Données démo générées aléatoirement mais réalistes

---
