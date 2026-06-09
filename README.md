# 🛒 Enterprise Retail Market Basket Analytics Dashboard

## 📌 Project Overview
A production-grade Market Basket Analysis engine designed to discover hidden purchasing patterns in retail transaction data. Built using the **Apriori algorithm**, this interactive web application allows business stakeholders to dynamically adjust thresholds and visualize product associations to drive data-informed retail decisions.

## 🚀 Key Features
* **Automated Data Pipeline:** Programmatically generates and processes realistic retail transaction datasets (1,000+ baskets).
* **Interactive Dashboard:** Built with Streamlit for real-time hyperparameter tuning (Support & Confidence).
* **Advanced Visualization:** Utilizes NetworkX & Matplotlib to plot high-confidence product pairings in an intuitive network graph.
* **Business-Centric Metrics:** Calculates exact Support, Confidence, and Lift metrics to identify strong item affinities.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas
* **Machine Learning:** Mlxtend (Apriori Algorithm)
* **Visualization:** NetworkX, Matplotlib
* **Front-end Framework:** Streamlit

## 💻 How to Run Locally

1. Clone the repository:
```bash
git clone [https://github.com/adarsh-0052/Retail-Market-Basket-Analytics.git](https://github.com/adarsh-0052/Retail-Market-Basket-Analytics.git)


2. Install the required dependencies:
```bash
pip install streamlit pandas mlxtend networkx matplotlib

3. Launch the Streamlit dashboard:
```bash
streamlit run app.py
