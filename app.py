import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =================================================================================
# 1. Load Pre-trained Models and Data
# =================================================================================

try:
    # Load the trained K-Means model and the scaler
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Load the product similarity matrix from the previous step
    similarity_df = pd.read_csv('product_similarity_matrix.csv', index_col=0)

except FileNotFoundError:
    st.error("Error: Model or data files not found. Please ensure 'kmeans_model.pkl', 'scaler.pkl', and 'product_similarity_matrix.csv' are in the same directory.")
    st.stop()


# =================================================================================
# 2. Helper Functions
# =================================================================================

# Create a mapping from cluster number to a descriptive label
cluster_labels = {
    0: "High-Value Shopper", # These labels are based on the cluster profiling
    1: "Big Spender",
    2: "At-Risk/Occasional",
    3: "Regular/Promising"
}

# Function to get product recommendations
def recommend_products(product_name, n_recommendations=5):
    """
    Finds and returns the top N similar products.
    """
    # Check if the product exists in our similarity matrix
    if product_name not in similarity_df.index:
        return "Product not found. Please try a different product name."

    # Get the similarity scores for the given product
    product_scores = similarity_df[product_name]

    # Sort the scores and get the top N recommendations (excluding the product itself)
    similar_products = product_scores.sort_values(ascending=False)
    recommended = similar_products.index[1:n_recommendations+1].tolist()
    return recommended


# =================================================================================
# 3. Streamlit UI and App Logic
# =================================================================================

st.set_page_config(
    page_title="Shopper Spectrum Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Product Recommender", "Customer Segmentation"])

if page == "Product Recommender":
    st.title("🛒 Product Recommender")
    st.markdown("Enter a product name to get 5 similar product recommendations.")
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/images/streamlit-logo-primary-colormark-light.png", width=200) # Placeholder image
    
    # Get user input for product name
    product_name_input = st.text_input("Enter Product Name (e.g., JUMBO BAG RED RETROSPOT)", "")
    
    if st.button("Get Recommendations"):
        if product_name_input:
            recommendations = recommend_products(product_name_input)
            
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                st.subheader("Recommended Products:")
                for i, product in enumerate(recommendations, 1):
                    st.write(f"{i}. {product}")
        else:
            st.warning("Please enter a product name.")

elif page == "Customer Segmentation":
    st.title("👤 Customer Segmentation")
    st.markdown("Enter RFM values to predict a customer's segment label.")
    
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/images/streamlit-logo-primary-colormark-light.png", width=200) # Placeholder image
    
    # Get user input for RFM values
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=100)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, max_value=5000, value=50)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, max_value=200000.0, value=500.0, format="%.2f")

    if st.button("Predict Segment"):
        # Create a DataFrame for the user's input
        input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        
        # Standardize the input data using the same scaler from training
        scaled_input = scaler.transform(input_data)
        
        # Predict the cluster
        predicted_cluster = kmeans.predict(scaled_input)[0]
        
        # Get the descriptive label from our mapping
        predicted_label = cluster_labels.get(predicted_cluster, "Unknown")
        
        st.subheader("Prediction Result:")
        st.success(f"This customer belongs to the **{predicted_label}** segment.")