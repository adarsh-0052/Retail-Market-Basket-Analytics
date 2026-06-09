import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import matplotlib.pyplot as plt
import os
import random

# Configure the dashboard layout
st.set_page_config(page_title="Retail Market Basket Analytics", layout="wide")

st.title("🛒 Enterprise Market Basket Analysis Dashboard")
st.markdown("A production-grade analytics engine processing transactional datasets to discover item associations.")
st.markdown("---")

# --- DATA PIPELINE: AUTOMATED DATA GENERATION & LOADING ---
CSV_FILE = "retail_transactions.csv"

# Function to generate a realistic large dataset if it doesn't exist
def generate_mock_dataset():
    items = ['Bread', 'Butter', 'Milk', 'Eggs', 'Diaper', 'Beer', 'Cola', 'Chips', 'Yogurt', 'Cheese']
    data = []
    
    # Simulating 1,000 unique customer transactions (Baskets)
    for transaction_id in range(1001, 2001):
        # Randomly decide how many items are in this basket (between 2 and 5)
        basket_size = random.randint(2, 5)
        
        # Injecting realistic purchasing patterns (Association Rules)
        prob = random.random()
        if prob < 0.40:
            # 40% chance the customer buys Bread and Butter together
            basket = ['Bread', 'Butter'] + random.sample([i for i in items if i not in ['Bread', 'Butter']], basket_size - 2)
        elif prob < 0.70:
            # 30% chance the customer buys Diaper and Beer together
            basket = ['Diaper', 'Beer'] + random.sample([i for i in items if i not in ['Diaper', 'Beer']], basket_size - 2)
        else:
            # 30% completely random shopping
            basket = random.sample(items, basket_size)
            
        for item in set(basket):
            data.append([transaction_id, item])
            
    df_generated = pd.DataFrame(data, columns=['Transaction_ID', 'Item_Name'])
    df_generated.to_csv(CSV_FILE, index=False)

# Automatically trigger dataset generation if file is missing
if not os.path.exists(CSV_FILE):
    generate_mock_dataset()

# Load the dataset from the CSV file
raw_data = pd.read_csv(CSV_FILE)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Model Hyperparameters")
st.sidebar.markdown("Tune thresholds for the Apriori Engine:")

min_support = st.sidebar.slider("Minimum Support (Item Popularity)", 0.05, 0.50, 0.15, 0.01)
min_confidence = st.sidebar.slider("Minimum Confidence (Association Strength)", 0.10, 1.00, 0.60, 0.05)

# --- BUSINESS METRICS SECTION (TOP CALLOUTS) ---
total_transactions = raw_data['Transaction_ID'].nunique()
unique_items_count = raw_data['Item_Name'].nunique()

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(label="Total Transactions Loaded", value=f"{total_transactions:,}")
with m2:
    st.metric(label="Unique Products Cataloged", value=unique_items_count)

# --- DATA TRANSFORMATION FOR MARKET BASKET ANALYSIS ---
# Group items by Transaction_ID to reconstruct individual shopping carts
basket_lists = raw_data.groupby('Transaction_ID')['Item_Name'].apply(list).tolist()

te = TransactionEncoder()
te_ary = te.fit(basket_lists).transform(basket_lists)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# --- EXECUTE APRIORI ALGORITHM ---
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    with m3:
        st.metric(label="Valid Association Rules Found", value=len(rules))
        
    if not rules.empty:
        # Clean set formatting for reporting
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        final_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        final_rules = final_rules.sort_values(by='confidence', ascending=False)
        
        # --- UI LAYOUT: SPLIT VIEW ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Association Rules Catalog")
            st.dataframe(final_rules, use_container_width=True, height=450)
            
        with col2:
            st.subheader("🕸️ Network Graph (Top 10 High-Confidence Links)")
            
            G = nx.DiGraph()
            top_rules = final_rules.head(10)
            
            for index, row in top_rules.iterrows():
                G.add_edge(row['antecedents'], row['consequents'], weight=row['confidence'])
                
            fig, ax = plt.subplots(figsize=(8, 6.2))
            pos = nx.spring_layout(G, k=0.6, seed=42)
            
            nx.draw_networkx_nodes(G, pos, node_size=1800, node_color='#34a853', edgecolors='black', ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='black', ax=ax)
            
            edges = G.edges()
            weights = [G[u][v]['weight'] * 4 for u,v in edges]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='#70757a', arrows=True, arrowsize=15, ax=ax)
            
            ax.axis('off')
            st.pyplot(fig)
    else:
        with m3: st.metric(label="Valid Association Rules Found", value=0)
        st.warning("No association rules found for the selected Confidence threshold.")
else:
    with m3: st.metric(label="Valid Association Rules Found", value=0)
    st.error("No frequent itemsets found for the selected Support threshold.")