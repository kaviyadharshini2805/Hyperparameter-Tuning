import streamlit as st
import pandas as pd
import joblib

# Load Model and Scaler
kmeans = joblib.load("kmeans_inventory_model.joblib")
scaler = joblib.load("inventory_scaler.joblib")

st.set_page_config(page_title="Supermarket Inventory Clustering", layout="centered")

# App Title
st.title("🛒 Supermarket Inventory Clustering")
st.write("This app predicts which **cluster/category** an inventory item belongs to using K-Means clustering.")

# User Inputs
st.header("📌 Enter Item Details")

stock_qty = st.number_input("Stock Quantity", min_value=0, step=1)
reorder_lvl = st.number_input("Reorder Level", min_value=0, step=1)
unit_price = st.number_input("Unit Price", min_value=0.0, step=0.1)

category = st.selectbox(
    "Category",
    ["Dairy", "Bakery", "Produce", "Snacks"]
)

supplier = st.selectbox(
    "Supplier",
    ["GreenFarm Produce", "Global Beverages", "Daily Dairy", "SnackWorld", "BakeHouse"]
)

# Encode Categories the same way as training
category_mapping = {
    "Dairy": 0,
    "Bakery": 1,
    "Produce": 2,
    "Snacks": 3
}

supplier_mapping = {
    "GreenFarm Produce": 0,
    "Global Beverages": 1,
    "Daily Dairy": 2,
    "SnackWorld": 3,
    "BakeHouse": 4
}

cat_code = category_mapping[category]
sup_code = supplier_mapping[supplier]

# Prediction Button
if st.button("🔍 Predict Cluster"):

    # Create input DataFrame
    input_data = pd.DataFrame([[stock_qty, reorder_lvl, unit_price, cat_code, sup_code]],
                              columns=["Stock Quantity", "Reorder Level", "Unit Price", "Category", "Supplier"])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]

    st.success(f"📦 This item belongs to **Cluster {cluster}**")

    # Interpretation
    st.subheader("📊 Cluster Interpretation")
    if cluster == 0:
        st.info("🟢 **High Stock – Low Reorder Risk**")
    elif cluster == 1:
        st.info("🟡 **Medium Stock – Moderate Reorder Risk**")
    elif cluster == 2:
        st.info("🔴 **Low Stock – High Reorder Priority**")

# Footer
st.write("---")
st.caption("Supermarket Inventory Clustering App • K-Means Machine Learning Model")
