# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.express as px

# 1. Chargement des données et modèles

@st.cache_data
def load_data():
    df_products = pd.read_csv("df_products.csv")
    df_events = pd.read_csv("data_clean.csv")  # events bruts: event_time, event_type, price, user_id, product_id, category_code, brand
    df_events["event_time"] = pd.to_datetime(df_events["event_time"])
    return df_products, df_events

@st.cache_resource
def load_models_and_encoders():
    user_encoder = joblib.load("encoders/user_encoder.pkl")
    product_encoder = joblib.load("encoders/product_encoder.pkl")
    price_scaler = joblib.load("encoders/price_scaler.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_users = len(user_encoder.classes_)
    num_products = len(product_encoder.classes_)

    class NCFModel(nn.Module):
        def __init__(self, num_users, num_items, emb_dim=32):
            super().__init__()
            self.user_emb = nn.Embedding(num_users, emb_dim)
            self.item_emb = nn.Embedding(num_items, emb_dim)
            input_dim = emb_dim * 2 + 1
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, user_idx, item_idx, price):
            u = self.user_emb(user_idx)
            i = self.item_emb(item_idx)
            x = torch.cat([u, i, price.unsqueeze(1)], dim=1)
            logit = self.mlp(x).squeeze(1)
            return logit

    model = NCFModel(num_users, num_products).to(device)
    model.load_state_dict(torch.load("models/ncf_model.pt", map_location=device))
    model.eval()

    return model, device, user_encoder, product_encoder, price_scaler



df_products, df_events = load_data()
model, device, user_encoder, product_encoder, price_scaler = load_models_and_encoders()

def get_mean_user_embedding(model):
    # moyenne des embeddings de tous les users connus
    with torch.no_grad():
        all_user_emb = model.user_emb.weight.data  # (num_users, emb_dim)
        mean_emb = all_user_emb.mean(dim=0, keepdim=True)  # (1, emb_dim)
    return mean_emb

mean_user_emb = get_mean_user_embedding(model)


def recommend_for_user(user_id_str, top_k=10):
    try:
        user_id = int(user_id_str)
    except:
        return pd.DataFrame(), []

    if user_id not in user_encoder.classes_:
        return pd.DataFrame(), []

    user_idx = user_encoder.transform([user_id])[0]

    # Produits déjà achetés (vérité terrain)
    user_purchases = df_events[
        (df_events["user_id"] == user_id) & (df_events["event_type"] == "purchase")
    ]["product_id"].unique().tolist()

    # Candidats: tous les produits
    prod_ids = df_products["product_id"].values
    prod_idx = product_encoder.transform(prod_ids)

    if "price_normalized" in df_products.columns:
        price_norm = df_products["price_normalized"].values
    else:
        prices = df_products["price"].values.reshape(-1, 1)
        price_norm = price_scaler.transform(prices).reshape(-1)

    u_tensor = torch.tensor([user_idx] * len(prod_idx), dtype=torch.long, device=device)
    p_tensor = torch.tensor(prod_idx, dtype=torch.long, device=device)
    price_tensor = torch.tensor(price_norm, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(u_tensor, p_tensor, price_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    rec_df = df_products.copy()
    rec_df["purchase_proba"] = probs
    rec_df = rec_df.sort_values("purchase_proba", ascending=False)

    top_recs = rec_df.head(top_k).copy()
    return top_recs, user_purchases

def recommend_for_new_profile(budget_min, budget_max, pref_category, pref_brand, top_k=10):
    # Filtrage des produits selon le profil
    df_cand = df_products.copy()
    df_cand = df_cand[(df_cand["price"] >= budget_min) & (df_cand["price"] <= budget_max)]

    if pref_category != "Toutes":
        df_cand = df_cand[df_cand["category_code"] == pref_category]
    if pref_brand != "Toutes":
        df_cand = df_cand[df_cand["brand"] == pref_brand]

    if df_cand.empty:
        return pd.DataFrame()

    # On utilise l'embedding moyen user comme user virtuel
    prod_ids = df_cand["product_id"].values
    prod_idx = product_encoder.transform(prod_ids)

    if "price_normalized" in df_cand.columns:
        price_norm = df_cand["price_normalized"].values
    else:
        prices = df_cand["price"].values.reshape(-1, 1)
        price_norm = price_scaler.transform(prices).reshape(-1)

    # Embedding moyen répété
    with torch.no_grad():
        item_emb = model.item_emb(torch.tensor(prod_idx, dtype=torch.long, device=device))  # (N, D)
        user_emb_rep = mean_user_emb.to(device).repeat(item_emb.size(0), 1)  # (N, D)
        price_tensor = torch.tensor(price_norm, dtype=torch.float32, device=device).unsqueeze(1)

        x = torch.cat([user_emb_rep, item_emb, price_tensor], dim=1)
        logits = model.mlp(x).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    rec_df = df_cand.copy()
    rec_df["purchase_proba"] = probs
    rec_df = rec_df.sort_values("purchase_proba", ascending=False).head(top_k)
    return rec_df



st.set_page_config(page_title="Reco-Cosmétiques", layout="wide")

# Exemple si le logo est à la racine du projet
LOGO_PATH = "logo.png"  # ou "assets/logo.png"

st.image(LOGO_PATH, use_container_width=True)

st.markdown(
    """
    <h1 style="text-align: center; margin-top: 0.5rem;">
        Système de recommandation – Cosmétiques
    </h1>
    """,
    unsafe_allow_html=True,
)

# Onglets principaux
tab_home, tab_user, tab_product, tab_new = st.tabs(["🏠 Vue globale", "🙋‍♀️ Recommandations utilisateur", "🧴 Analyse produits", "🆕 Nouveau client"])

# 1) Vue globale
with tab_home:
    st.subheader("Statistiques globales")

    # Filtres
    min_date = df_events["event_time"].min().date()
    max_date = df_events["event_time"].max().date()
    col_f1, col_f2 = st.columns(2)
    date_range = col_f1.date_input("Période", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    category_filter = col_f2.selectbox(
        "Catégorie (category_code)",
        options=["Toutes"] + sorted(df_events["category_code"].dropna().unique().tolist()),
    )

    start_date, end_date = date_range
    mask = (df_events["event_time"].dt.date >= start_date) & (df_events["event_time"].dt.date <= end_date)
    if category_filter != "Toutes":
        mask &= df_events["category_code"] == category_filter
    df_filt = df_events[mask]
    
    # KPIs de base
    n_users = df_filt["user_id"].nunique()
    n_products = df_filt["product_id"].nunique()
    n_events = len(df_filt)
    n_views = (df_filt["event_type"] == "view").sum()
    n_cart = (df_filt["event_type"] == "cart").sum()
    n_purch = (df_filt["event_type"] == "purchase").sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Utilisateurs uniques", f"{n_users:,}")
    col2.metric("Produits uniques", f"{n_products:,}")
    col3.metric("Événements totaux", f"{n_events:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Vues", f"{n_views:,}")
    col5.metric("Ajouts panier", f"{n_cart:,}")
    col6.metric("Achats", f"{n_purch:,}")

    # Taux de conversion
    conv_view_to_purchase = n_purch / max(n_views, 1)
    conv_cart_to_purchase = n_purch / max(n_cart, 1)

    col7, col8 = st.columns(2)
    col7.metric("Conversion vues → achats", f"{conv_view_to_purchase*100:.2f} %")
    col8.metric("Conversion panier → achats", f"{conv_cart_to_purchase*100:.2f} %")

        # Répartition des ventes par catégorie / marque
    df_purch = df_filt[df_filt["event_type"] == "purchase"].copy()
    df_purch["revenue"] = df_purch["price"]

    # Bar chart par catégorie
    if not df_purch.empty:
        cat_rev = (
            df_purch.groupby("category_code")["revenue"]
            .sum()
            .reset_index()
            .sort_values("revenue", ascending=False)
            .head(20)
        )
        fig_cat = px.bar(cat_rev, x="category_code", y="revenue", title="Revenu par catégorie (Top 20)")
        fig_cat.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cat, use_container_width=True)

        # Treemap par marque
        brand_rev = (
            df_purch.groupby(["category_code", "brand"])["revenue"]
            .sum()
            .reset_index()
        )
        fig_tree = px.treemap(
            brand_rev,
            path=["category_code", "brand"],
            values="revenue",
            title="Répartition des ventes par catégorie / marque",
        )
        st.plotly_chart(fig_tree, use_container_width=True)

        # CA mensuel
        df_purch["month"] = df_purch["event_time"].dt.to_period("M").dt.to_timestamp()
        rev_month = df_purch.groupby("month")["revenue"].sum().reset_index()
        fig_rev = px.line(rev_month, x="month", y="revenue", title="Évolution du chiffre d’affaires mensuel")
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("Pas d'achats dans la période / catégorie sélectionnée.")


# 2) Recommandations utilisateur
with tab_user:
    st.subheader("Recommandations personnalisées")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        user_input = st.text_input("User ID", value="463240011")
        top_k = st.slider("Top K", min_value=5, max_value=30, value=10, step=5)

        bouton = st.button("Générer des recommandations")

    if bouton:
        recs, user_purchases = recommend_for_user(user_input, top_k=top_k)
        if recs.empty:
            st.warning("Utilisateur inconnu ou pas assez de données.")
        else:
            st.success(f"Top {top_k} produits recommandés pour l'utilisateur {user_input}")

            cols_show = ["product_id", "brand", "category_code", "price", "purchase_proba"]
            cols_show = [c for c in cols_show if c in recs.columns]
            
            # Tableau à gauche
            with col_left:
                st.dataframe(recs[cols_show])

            # Graphiques à droite
            with col_right:
                # Bar chart des probas
                fig_bar = px.bar(
                    recs,
                    x="product_id",
                    y="purchase_proba",
                    title="Probabilité d'achat pour les produits recommandés",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Histogramme des probas (Top 20)
                top20 = recs.head(20)
                fig_hist = px.histogram(
                    top20,
                    x="purchase_proba",
                    nbins=10,
                    title="Distribution des probabilités d'achat (Top 20)",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Precision@K / Recall@K sous les deux colonnes
            if user_purchases:
                rec_ids = recs["product_id"].tolist()
                act_set = set(user_purchases)
                pred_set = set(rec_ids)

                intersection = act_set & pred_set
                precision_k = len(intersection) / float(top_k)
                recall_k = len(intersection) / float(len(act_set))

                st.markdown(f"**Produits Total achetés par l’utilisateur:** {len(act_set)}")
                st.markdown(f"**Produits achetés dans le Top {top_k}:** {len(intersection)}")
                #st.metric(f"Precision@{top_k}", f"{precision_k:.2f}")
                #st.metric(f"Recall@{top_k}", f"{recall_k:.2f}")
            else:
                st.info("Cet utilisateur n'a pas d'achats historiques.")



# 3) Analyse produits
with tab_product:
    st.subheader("Exploration des produits")

    # Filtres
    selected_brand = st.selectbox(
        "Marque",
        options=["Toutes"] + sorted(df_products["brand"].dropna().unique().tolist()),
    )
    selected_category = st.selectbox(
        "Catégorie",
        options=["Toutes"] + sorted(df_products["category_code"].dropna().unique().tolist()),
    )

    df_filtered = df_products.copy()
    if selected_brand != "Toutes":
        df_filtered = df_filtered[df_filtered["brand"] == selected_brand]
    if selected_category != "Toutes":
        df_filtered = df_filtered[df_filtered["category_code"] == selected_category]

    st.write(f"{len(df_filtered)} produits pour la sélection actuelle.")

    # Top produits par vues / achats / conversion
    cols_top = st.multiselect(
        "Critères pour les produits stars",
        options=["prod_total_views", "prod_total_purchases", "prod_conversion_rate"],
        default=["prod_total_purchases"],
    )

    for col in cols_top:
        if col in df_filtered.columns:
            top_star = df_filtered.sort_values(col, ascending=False).head(10)
            st.markdown(f"**Top 10 produits par `{col}`**")
            st.dataframe(top_star[["product_id", "brand", "category_code", "price", col]])

    # Scatter popularité vs conversion
    metric_x = st.selectbox(
        "Axe X (popularité)",
        options=["prod_total_views", "prod_total_purchases"],
        index=0,
    )
    metric_y = st.selectbox(
        "Axe Y (performance)",
        options=["prod_conversion_rate"],
        index=0,
    )

    if {"prod_total_views", "prod_total_purchases", "prod_conversion_rate"}.issubset(df_filtered.columns):
        fig_seg = px.scatter(
            df_filtered,
            x=metric_x,
            y=metric_y,
            color="brand" if selected_brand == "Toutes" else "category_code",
            size="price",
            hover_name="product_id",
            title=f"{metric_y} vs {metric_x}",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    # Analyse prix
    st.subheader("Analyse prix")

    # On travaille sur une copie pour éviter de modifier df_products global
    df_price = df_products.copy()

    if "purchase_proba_mean" in df_price.columns:
        threshold = df_price["purchase_proba_mean"].median()
        df_price["is_recommended_like"] = (df_price["purchase_proba_mean"] >= threshold).astype(int)
    else:
        threshold = df_price["prod_conversion_rate"].median()
        df_price["is_recommended_like"] = (df_price["prod_conversion_rate"] >= threshold).astype(int)

    df_rec = df_price[df_price["is_recommended_like"] == 1]
    df_non = df_price[df_price["is_recommended_like"] == 0]

    fig_hist_price = px.histogram(
    df_rec,
    x="price",
    nbins=40,  # OK ici
    opacity=0.6,
    marginal="box",
    title="Prix des produits 'recommandés-like'",
)

    fig_hist_price.add_histogram(
        x=df_non["price"],
        nbinsx=40,          # <-- correction ici
        opacity=0.4,
        name="Produits non recommandés-like",
    )

    st.plotly_chart(fig_hist_price, use_container_width=True)


    fig_price_corr = px.scatter(
        df_price,
        x="price",
        y="prod_conversion_rate",
        color="brand" if selected_brand == "Toutes" else "category_code",
        title="Prix vs taux de conversion",
        hover_name="product_id",
    )
    st.plotly_chart(fig_price_corr, use_container_width=True)

# 4) Nouveau client (cold-start)
with tab_new:
    st.subheader("Simulation pour un nouveau client")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### Profil du client")

        age = st.number_input("Âge (optionnel)", min_value=15, max_value=90, value=30, step=1)
        gender = st.selectbox("Genre (optionnel)", ["Non spécifié", "Femme", "Homme"])

        budget_min, budget_max = st.slider(
            "Budget approximatif (€)",
            min_value=float(df_products["price"].min()),
            max_value=float(df_products["price"].max()),
            value=(
                float(df_products["price"].quantile(0.25)),
                float(df_products["price"].quantile(0.75)),
            ),
        )

        pref_category = st.selectbox(
            "Catégorie préférée",
            options=["Toutes"] + sorted(df_products["category_code"].dropna().unique().tolist()),
        )

        pref_brand = st.selectbox(
            "Marque préférée (optionnel)",
            options=["Toutes"] + sorted(df_products["brand"].dropna().unique().tolist()),
        )

        top_k_new = st.slider("Nombre de recommandations", min_value=5, max_value=30, value=10, step=5)

        bouton_new = st.button("Générer des recommandations pour ce profil")

    if bouton_new:
        recs_new = recommend_for_new_profile(
            budget_min, budget_max, pref_category, pref_brand, top_k=top_k_new
        )

        if recs_new.empty:
            st.warning("Aucun produit ne correspond à ce profil (filtre trop restrictif).")
        else:
            st.success(f"Top {top_k_new} produits recommandés pour ce nouveau profil")

            # Tableau à gauche
            cols_show_new = ["product_id", "brand", "category_code", "price", "purchase_proba"]
            cols_show_new = [c for c in cols_show_new if c in recs_new.columns]
            col_left.dataframe(recs_new[cols_show_new])

            # Visualisation des probabilités à droite
            fig_bar_new = px.bar(
                recs_new,
                x="product_id",
                y="purchase_proba",
                title="Probabilité d'achat (profil nouveau client)",
            )
            col_right.plotly_chart(fig_bar_new, use_container_width=True)





#streamlit run streamlit_app.py