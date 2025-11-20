import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import json
from datetime import datetime, timedelta
import warnings
import time

# Ignorer les warnings pour une UI propre
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONFIGURATION GLOBALE ET PAGE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexus Analytics Pro",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. MOTEUR DE STYLE CSS (DESIGN SYSTEM)
# -----------------------------------------------------------------------------
# Ce bloc définit l'identité visuelle complète de l'application
st.markdown("""
    <style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');

    /* VARIABLES DE COULEURS */
    :root {
        --bg-color: #FFFFFFF;
        --sidebar-bg: #2C3E50; /* Bleu nuit profond */
        --card-bg: #D9C8A0;
        --primary: #3498DB;
        --secondary: #9B59B6;
        --success: #2ECC71;
        --warning: #F1C40F;
        --danger: #E74C3C;
        --text-dark: #2C3E50;
        --text-light: #000000;
        --card-radius: 16px;
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
        --shadow-md: 0 8px 24px rgba(0,0,0,0.08);
    }

    /* GLOBAL RESET */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span {
        color: #ECF0F1 !important;
    }
    
    /* NAVIGATION RADIO BUTTONS */
    .stRadio > div {
        gap: 10px;
    }
    .stRadio > div > label {
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 12px 15px;
        color: white !important;
        transition: all 0.3s ease;
    }
    .stRadio > div > label:hover {
        background-color: rgba(255,255,255,0.15) !important;
        transform: translateX(5px);
    }
    .stRadio > div > [data-baseweb="radio"] > div:first-child {
        background-color: var(--primary) !important;
        border-color: var(--primary) !important;
    }

    /* CARTE DESIGN (GLASSMORPHISM LITE) */
    .nexus-card {
        background-color: var(--card-bg);
        border-radius: var(--card-radius);
        padding: 24px;
        box-shadow: var(--shadow-sm);
        border: 1px solid rgba(255,255,255,0.5);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .nexus-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-md);
    }

    /* TYPOGRAPHIE INTERNE AUX CARTES */
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 15px;
        font-weight: 600;
        color: var(--text-light);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--text-dark);
        font-family: 'Poppins', sans-serif;
        margin: 5px 0;
    }
    .metric-delta {
        font-size: 13px;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        border-radius: 20px;
    }
    .delta-pos { background: #E8F8F5; color: var(--success); }
    .delta-neg { background: #FDEFEF; color: var(--danger); }

    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: white;
        padding: 10px 20px;
        border-radius: 50px;
        box-shadow: var(--shadow-sm);
        width: fit-content;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 20px;
        padding: 0 20px;
        background-color: transparent;
        border: none;
        color: var(--text-light);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white !important;
    }

    /* INPUTS & WIDGETS */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 12px;
        border: 1px solid #E0E0E0;
    }
    .stNumberInput input {
        border-radius: 12px;
    }

    /* UTILS */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #E0E0E0, transparent);
        margin: 20px 0;
    }
    
    /* ANIMATIONS */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. GESTION DES DONNÉES (ROBUSTE ET GÉNÉRATEUR)
# -----------------------------------------------------------------------------

@st.cache_data
def generate_dummy_data():
    """Génère des données si aucun fichier n'est trouvé pour la démo"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    categories = ['Electronics', 'Furniture', 'Office Supplies', 'Software']
    
    data = []
    for date in dates:
        n_transactions = np.random.randint(5, 15)
        for _ in range(n_transactions):
            price = np.random.uniform(50, 2000)
            qty = np.random.randint(1, 10)
            discount = np.random.choice([0, 0.05, 0.1, 0.2], p=[0.6, 0.2, 0.15, 0.05])
            
            data.append({
                'Sale_Date': date,
                'Region': np.random.choice(regions),
                'Product_Category': np.random.choice(categories),
                'Sales_Amount': price * qty * (1 - discount),
                'Quantity_Sold': qty,
                'Unit_Price': price,
                'Discount': discount,
                'Profi': (price * qty * (1 - discount)) * np.random.uniform(0.1, 0.4), # Profi ~10-40%
                'Sales_Rep': f"Rep_{np.random.randint(1, 20)}"
            })
    
    df = pd.DataFrame(data)
    df['Region_and_Sales_Rep'] = df['Region'] + " - " + df['Sales_Rep']
    return df

@st.cache_data
def load_data():
    """Charge les données réelles ou génère des fausses"""
    data_path = Path('output/data/cleaned_sales_data.csv')
    if data_path.exists():
        try:
            df = pd.read_csv(data_path, parse_dates=['Sale_Date'])
            return df
        except Exception as e:
            st.warning(f"Erreur chargement fichier: {e}. Utilisation données démo.")
            return generate_dummy_data()
    else:
        # Si pas de fichier, on génère silencieusement des données pour la démo
        return generate_dummy_data()

# Chargement initial
df = load_data()

# Calculs globaux pour réutilisation
total_sales = df['Sales_Amount'].sum()
total_Profi = df['Profi'].sum()
avg_margin = (total_Profi / total_sales) * 100
current_date = df['Sale_Date'].max()

# -----------------------------------------------------------------------------
# 4. COMPOSANTS UI RÉUTILISABLES
# -----------------------------------------------------------------------------

def card_metric(title, value, delta=None, prefix="", suffix="", color="text-dark"):
    """Affiche une carte métrique stylisée"""
    delta_html = ""
    if delta is not None:
        delta_cls = "delta-pos" if delta >= 0 else "delta-neg"
        icon = "▲" if delta >= 0 else "▼"
        delta_html = f'<span class="metric-delta {delta_cls}">{icon} {abs(delta)}%</span>'
    
    st.markdown(f"""
    <div class="nexus-card animate-fade-in">
        <div class="card-header">
            <span class="card-title">{title}</span>
            <span style="font-size: 18px;">📊</span>
        </div>
        <div class="metric-value" style="color: var(--{color})">
            {prefix}{value}{suffix}
        </div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def card_chart_wrapper(title, chart_func, height=300):
    """Enveloppe un graphique Plotly dans le style 'Card'"""
    st.markdown(f"""<div class="nexus-card animate-fade-in">
        <div class="card-title" style="margin-bottom: 15px;">{title}</div>
    """, unsafe_allow_html=True)
    chart_func(height=height)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. NAVIGATION SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #3498DB, #9B59B6); 
                        border-radius: 50%; margin: 0 auto 10px auto; display: flex; align-items: center; justify-content: center; font-size: 30px;">
                💠
            </div>
            <h2 style="color: white; margin: 0; font-size: 20px;">NEXUS</h2>
            <p style="color: #95A5A6; font-size: 12px; letter-spacing: 2px;">ANALYTICS PRO</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    nav_selection = st.radio(
        "NAVIGATION",
        [
            "🏠 Tableau de Bord", 
            "📉 Analyse Détaillée", 
            "🗺️ Géographie & Segments", 
            "🔮 Simulateur IA", 
            "🤖 Machine Learning", 
            "📑 Rapports & Données"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Sidebar Widgets
    st.markdown("<p style='font-size:12px; text-transform:uppercase; letter-spacing:1px; color:#95A5A6; margin-bottom:10px;'>Filtres Rapides</p>", unsafe_allow_html=True)
    
    selected_region = st.multiselect("Régions", options=df['Region'].unique(), default=df['Region'].unique())
    selected_cat = st.multiselect("Catégories", options=df['Product_Category'].unique(), default=df['Product_Category'].unique())
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Carte info utilisateur
    st.markdown("""
        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; display: flex; align-items: center; gap: 10px;">
            <div style="width: 35px; height: 35px; background: #3498DB; border-radius: 50%;"></div>
            <div>
                <div style="font-size: 13px; font-weight: 600; color: white;">Admin User</div>
                <div style="font-size: 11px; color: #95A5A6;">Premium Plan</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Filtrage des données
df_filtered = df[df['Region'].isin(selected_region) & df['Product_Category'].isin(selected_cat)]

# -----------------------------------------------------------------------------
# 6. CONTENU DES PAGES
# -----------------------------------------------------------------------------

# HEADER COMMUN
st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px;">
        <div>
            <h1 style="margin:0; font-size: 28px;">{nav_selection.split(' ')[1]}</h1>
            <p style="margin:0; color: #7F8C8D;">Dernière mise à jour : {datetime.now().strftime('%d %B %Y')} • <span style="color:var(--success)">● En ligne</span></p>
        </div>
        <div style="background: white; padding: 10px 20px; border-radius: 30px; box-shadow: var(--shadow-sm); display: flex; align-items: center; gap: 10px;">
            <span style="color: #BDC3C7;">🔍</span>
            <span style="color: #7F8C8D; font-size: 14px;">Recherche globale...</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if df_filtered.empty:
    st.error("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

# =============================================================================
# PAGE 1: TABLEAU DE BORD (DASHBOARD)
# =============================================================================
if "Tableau de Bord" in nav_selection:
    
    # --- Ligne 1: KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculs dynamiques (comparaison vs période précédente simulée)
    with col1:
        card_metric("Chiffre d'Affaires", f"{df_filtered['Sales_Amount'].sum()/1000:,.1f}k", 12.5, prefix="$")
    with col2:
        card_metric("Commandes", f"{len(df_filtered):,}", -2.4)
    with col3:
        Profi = df_filtered['Profi'].sum()
        margin = (Profi / df_filtered['Sales_Amount'].sum()) * 100
        card_metric("Marge Nette", f"{margin:.1f}", 5.3, suffix="%")
    with col4:
        avg_basket = df_filtered['Sales_Amount'].mean()
        card_metric("Panier Moyen", f"{avg_basket:.0f}", 0.8, prefix="$")
    
    # --- Ligne 2: Graphique Principal + Top Produits ---
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        def plot_sales_trend(height):
            daily = df_filtered.groupby('Sale_Date')[['Sales_Amount', 'Profi']].sum().reset_index()
            daily['MA7'] = daily['Sales_Amount'].rolling(7).mean()
            
            fig = go.Figure()
            # Zone de fond (Sales)
            fig.add_trace(go.Scatter(
                x=daily['Sale_Date'], y=daily['Sales_Amount'],
                mode='lines', fill='tozeroy', name='Ventes',
                line=dict(color='#3498DB', width=1),
                fillcolor='rgba(52, 152, 219, 0.1)'
            ))
            # Ligne de tendance (Profi)
            fig.add_trace(go.Scatter(
                x=daily['Sale_Date'], y=daily['Profi'],
                mode='lines', name='Profi',
                line=dict(color='#2ECC71', width=2)
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#F0F2F5'),
                legend=dict(orientation="h", y=1.1),
                height=height
            )
            st.plotly_chart(fig, use_container_width=True)
            
        card_chart_wrapper("Évolution Ventes & Profi (YTD)", plot_sales_trend, height=380)

    with col_side:
        # Liste stylisée des meilleures catégories
        st.markdown('<div class="nexus-card animate-fade-in" style="height: 456px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Performance Catégories</div>', unsafe_allow_html=True)
        
        cat_perf = df_filtered.groupby('Product_Category')['Sales_Amount'].sum().sort_values(ascending=False)
        max_val = cat_perf.max()
        
        for cat, val in cat_perf.items():
            pct = (val / max_val) * 100
            color = "#3498DB" if pct > 75 else "#9B59B6" if pct > 40 else "#95A5A6"
            
            st.markdown(f"""
                <div style="margin-bottom: 20px;">
                    <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; font-weight:500;">
                        <span>{cat}</span>
                        <span style="color:{color}">${val/1000:.0f}k</span>
                    </div>
                    <div style="width:100%; background:#F0F2F5; height:6px; border-radius:3px;">
                        <div style="width:{pct}%; background:{color}; height:6px; border-radius:3px; transition: width 1s ease;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Ligne 3: Répartition ---
    c1, c2 = st.columns(2)
    
    with c1:
        def plot_donut(height):
            fig = px.pie(df_filtered, names='Region', values='Sales_Amount', hole=0.6,
                         color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(showlegend=True, margin=dict(l=20, r=0, t=0, b=0), height=height)
            fig.update_traces(textinfo='percent+label', textposition='inside')
            st.plotly_chart(fig, use_container_width=True)
        card_chart_wrapper("Répartition Géographique", plot_donut, height=300)
        
    with c2:
        def plot_bar_stack(height):
            # Top 5 Sales Reps
            top_reps = df_filtered.groupby('Region_and_Sales_Rep')['Sales_Amount'].sum().nlargest(5).index
            df_top = df_filtered[df_filtered['Region_and_Sales_Rep'].isin(top_reps)]
            
            fig = px.bar(df_top, x='Region_and_Sales_Rep', y='Sales_Amount', color='Product_Category',
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False,
                xaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True)
        card_chart_wrapper("Top 5 Vendeurs par Mix Produit", plot_bar_stack, height=300)

# =============================================================================
# PAGE 2: ANALYSE DÉTAILLÉE (DEEP DIVE)
# =============================================================================
elif "Analyse Détaillée" in nav_selection:
    
    tabs = st.tabs(["📊 Distributions", "🌡️ Corrélations", "📅 Saisonnalité"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            card_chart_wrapper("Distribution des Prix Unitaires", 
                               lambda height: st.plotly_chart(px.histogram(df_filtered, x="Unit_Price", nbins=30, color_discrete_sequence=['#3498DB']).update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True))
        with col2:
            card_chart_wrapper("Distribution des Profis", 
                               lambda height: st.plotly_chart(px.box(df_filtered, x="Product_Category", y="Profi", color="Product_Category").update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'), use_container_width=True))
    
    with tabs[1]:
        # Heatmap de corrélation
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Matrice de Corrélation</div>', unsafe_allow_html=True)
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        corr = df_filtered[numeric_cols].corr()
        
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Scatter Plot interactif
        col_x, col_y, col_c = st.columns(3)
        with col_x: x_axis = st.selectbox("Axe X", numeric_cols, index=0)
        with col_y: y_axis = st.selectbox("Axe Y", numeric_cols, index=4) # Profi default
        with col_c: color_var = st.selectbox("Couleur", ['Region', 'Product_Category'])
        
        fig = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_var, size='Quantity_Sold', 
                         hover_data=['Region_and_Sales_Rep'], template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        # Analyse temporelle (Heatmap calendrier)
        df_filtered['Month'] = df_filtered['Sale_Date'].dt.month_name()
        df_filtered['Day'] = df_filtered['Sale_Date'].dt.day_name()
        
        pivot_hm = df_filtered.pivot_table(index='Day', columns='Month', values='Sales_Amount', aggfunc='sum').fillna(0)
        # Ordonner les jours
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_hm = pivot_hm.reindex(days_order)
        
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Intensité des Ventes: Jour vs Mois</div>', unsafe_allow_html=True)
        fig = px.imshow(pivot_hm, labels=dict(x="Mois", y="Jour", color="Ventes"), color_continuous_scale="Viridis")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# PAGE 3: GÉOGRAPHIE ET SEGMENTATION (PARETO)
# =============================================================================
elif "Géographie & Segments" in nav_selection:
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Analyse Pareto (Loi des 80/20)")
        st.caption("Identifiez les produits qui génèrent 80% de votre chiffre d'affaires.")
        
        # Préparation Pareto
        pareto_df = df_filtered.groupby('Region_and_Sales_Rep')['Sales_Amount'].sum().sort_values(ascending=False).reset_index()
        pareto_df['Cumulative_Sales'] = pareto_df['Sales_Amount'].cumsum()
        pareto_df['Cumulative_Pct'] = pareto_df['Cumulative_Sales'] / pareto_df['Sales_Amount'].sum() * 100
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Bar(x=pareto_df['Region_and_Sales_Rep'], y=pareto_df['Sales_Amount'], name="Ventes", marker_color="#3498DB"), secondary_y=False)
        fig.add_trace(go.Scatter(x=pareto_df['Region_and_Sales_Rep'], y=pareto_df['Cumulative_Pct'], name="Cumul %", marker_color="#E74C3C", mode="lines"), secondary_y=True)
        
        fig.update_layout(height=500, title_text="Pareto des Vendeurs")
        fig.update_yaxes(title_text="Montant (€)", secondary_y=False)
        fig.update_yaxes(title_text="Cumul (%)", secondary_y=True, range=[0, 110])
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Treemap des Catégories")
        st.caption("Vue hiérarchique Région > Catégorie")
        
        fig_tree = px.treemap(df_filtered, path=['Region', 'Product_Category'], values='Sales_Amount',
                              color='Profi', color_continuous_scale='RdBu')
        fig_tree.update_layout(height=500)
        st.plotly_chart(fig_tree, use_container_width=True)
        
    # Sunburst Chart
    st.markdown("---")
    st.subheader("Vue Radiale des Ventes")
    fig_sun = px.sunburst(df_filtered, path=['Region', 'Product_Category', 'Region_and_Sales_Rep'], values='Sales_Amount', color='Profi')
    fig_sun.update_layout(height=600)
    st.plotly_chart(fig_sun, use_container_width=True)

# =============================================================================
# PAGE 4: SIMULATEUR AVANCÉ (WHAT-IF)
# =============================================================================
elif "Simulateur IA" in nav_selection:
    st.markdown("""
    <div class="nexus-card">
        <h3 style="color:var(--primary)">🔮 Simulateur de Scénarios</h3>
        <p>Modifiez les paramètres ci-dessous pour voir l'impact projeté sur la marge et le Profi global.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_params, col_res = st.columns([1, 2])
    
    with col_params:
        st.markdown("#### Paramètres d'entrée")
        sim_price_change = st.slider("Variation Prix (%)", -20, 20, 0, help="Impact sur le prix unitaire")
        sim_vol_change = st.slider("Variation Volume (%)", -20, 20, 0, help="Impact sur la quantité vendue")
        sim_cost_change = st.slider("Variation Coûts (%)", -10, 10, 0, help="Inflation des coûts fournisseurs")
        
        st.markdown("#### Hypothèses Elasticité")
        elasticity = st.number_input("Élasticité Prix", value=-1.5, step=0.1, help="Si prix +1%, Volume change de X%")
        
        # Calcul automatique de l'impact volume si prix change (basé sur élasticité)
        if sim_price_change != 0:
            implied_vol_change = sim_price_change * elasticity
            st.info(f"L'élasticité suggère un impact volume de {implied_vol_change:.1f}%")
    
    with col_res:
        # Logique de simulation
        base_sales = df_filtered['Sales_Amount'].sum()
        base_Profi = df_filtered['Profi'].sum()
        base_cost = base_sales - base_Profi
        
        # Application scénario
        new_sales_vol_factor = 1 + ((sim_vol_change + (sim_price_change * elasticity))/100)
        new_price_factor = 1 + (sim_price_change/100)
        new_cost_factor = 1 + (sim_cost_change/100)
        
        projected_sales = base_sales * new_sales_vol_factor * new_price_factor
        projected_costs = base_cost * new_sales_vol_factor * new_cost_factor
        projected_Profi = projected_sales - projected_costs
        
        # Affichage résultats
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Sales Projetés", f"${projected_sales:,.0f}", f"{(projected_sales/base_sales - 1)*100:.1f}%")
        with c2:
            st.metric("Coûts Projetés", f"${projected_costs:,.0f}", f"{(projected_costs/base_cost - 1)*100:.1f}%", delta_color="inverse")
        with c3:
            st.metric("Profi Projeté", f"${projected_Profi:,.0f}", f"{(projected_Profi/base_Profi - 1)*100:.1f}%")
            
        # Graphique Waterfall
        fig = go.Figure(go.Waterfall(
            name = "20", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Base Profi", "Impact Prix/Vol", "Impact Coûts", "Nouveau Profi"],
            textposition = "outside",
            text = [f"{base_Profi/1000:.0f}k", "", "", f"{projected_Profi/1000:.0f}k"],
            y = [base_Profi, projected_sales - base_sales - (projected_costs - base_cost) + (projected_costs - base_cost), -(projected_costs - base_cost), projected_Profi],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title = "Analyse d'impact du scénario (Waterfall)", height=400)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE 5: MACHINE LEARNING (SUMMARY)
# =============================================================================
elif "Machine Learning" in nav_selection:
    st.subheader("🤖 Centre de Modélisation Prédictive")
    
    # Simulation de chargement de modèle si les fichiers n'existent pas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Performance des Modèles (Benchmarking)</div>', unsafe_allow_html=True)
        
        # Données factices pour le graphique si pas de JSON
        models_data = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost', 'Linear Regression', 'SVR'],
            'R2 Score': [0.89, 0.92, 0.75, 0.81],
            'MAE': [120, 95, 210, 150],
            'Training Time (s)': [4.5, 6.2, 0.5, 12.1]
        })
        
        fig = px.bar(models_data, y='Model', x='R2 Score', color='R2 Score', 
                     orientation='h', color_continuous_scale='Viridis', text_auto=True)
        fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Importance (Simulée)
        st.markdown('<div class="nexus-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Importance des Variables (Feature Importance)</div>', unsafe_allow_html=True)
        feat_imp = pd.DataFrame({
            'Feature': ['Discount', 'Unit_Price', 'Region_Europe', 'Category_Tech', 'Month', 'DayOfWeek'],
            'Importance': [0.45, 0.30, 0.10, 0.08, 0.05, 0.02]
        })
        fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig_imp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="nexus-card" style="background: #2C3E50; color: white;">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:white">Modèle Actif</h3>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:40px; font-weight:bold; color:#2ECC71">XGBoost</div>', unsafe_allow_html=True)
        st.markdown("v2.1.0 • Entraîné hier")
        st.markdown("---")
        st.markdown("**Précision Globale:** 92%")
        st.markdown("**Erreur Moyenne:** $95")
        st.button("🔄 Ré-entraîner", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.info("Le modèle détecte que 'Discount' est le facteur prédictif #1 pour le volume de ventes ce mois-ci.")

# =============================================================================
# PAGE 6: RAPPORTS ET DONNÉES
# =============================================================================
elif "Rapports & Données" in nav_selection:
    st.subheader("📑 Gestion des Données")
    
    with st.expander("Visualiser les données brutes", expanded=True):
        st.dataframe(df_filtered, use_container_width=True, height=400)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Télécharger CSV (Filtré)",
            data=df_filtered.to_csv(index=False).encode('utf-8'),
            file_name='nexus_export_data.csv',
            mime='text/csv',
            type='primary'
        )
    with col2:
        st.download_button(
            label="📄 Générer Rapport PDF (Simulé)",
            data=b"PDF Content",
            file_name='rapport_mensuel.pdf',
            disabled=True,
            help="Fonctionnalité disponible dans la version Enterprise"
        )
        
    # Section Logs système
    st.markdown("### 🛠️ Logs Système")
    logs = pd.DataFrame({
        'Timestamp': [datetime.now() - timedelta(minutes=i*15) for i in range(5)],
        'Event': ['Data Refresh', 'Model Inference', 'User Login', 'Export CSV', 'System Check'],
        'Status': ['Success', 'Success', 'Success', 'Warning', 'Success'],
        'User': ['System', 'API', 'Admin', 'Admin', 'System']
    })
    st.table(logs)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
    <div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #E0E0E0; color: #95A5A6; font-size: 12px;">
        © 2025 Nexus Analytics Corporation. All rights reserved.<br>
        Designed with ❤️ using Streamlit & Plotly.
    </div>
""", unsafe_allow_html=True)