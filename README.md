### Shopper Spectrum: Customer Segmentation and Product Recommendations in E-Commerce

## 📝 Project Overview

[cite\_start]The global e-commerce industry generates vast amounts of transaction data daily, which offers valuable insights into customer purchasing behaviors[cite: 3]. [cite\_start]Analyzing this data is essential for identifying meaningful customer segments and recommending relevant products to enhance customer experience and drive business growth[cite: 4].

[cite\_start]This project aims to address this challenge by examining transactional data from an online retail business to uncover patterns in customer behavior[cite: 5]. The core of this project involves two main objectives:

1.  [cite\_start]**Customer Segmentation**: Segmenting customers based on Recency, Frequency, and Monetary (RFM) analysis[cite: 5].
2.  [cite\_start]**Product Recommendation**: Developing a product recommendation system using item-based collaborative filtering techniques[cite: 5, 49].

## 🎯 Business Use Cases

[cite\_start]The insights and models developed in this project have several real-world business applications[cite: 6]:

  * [cite\_start]**Targeted Marketing Campaigns**: Creating customized marketing campaigns for different customer segments[cite: 7].
  * [cite\_start]**Personalized Product Recommendations**: Providing personalized product recommendations on e-commerce platforms[cite: 8].
  * [cite\_start]**Customer Retention**: Identifying "At-Risk Customers" for retention programs[cite: 9].
  * [cite\_start]**Inventory Management**: Optimizing inventory and stock based on customer demand patterns[cite: 11].

## 🧠 Project Methodology

The project follows a structured data science workflow, from data preprocessing to model deployment.

1.  [cite\_start]**Data Preprocessing**: The raw transaction data was cleaned by removing rows with missing `CustomerID`, excluding canceled invoices, and filtering out negative or zero quantities and prices[cite: 23, 24, 25].
2.  [cite\_start]**Exploratory Data Analysis (EDA)**: We performed a thorough analysis to understand transaction volume by country, identify top-selling products, and inspect monetary distributions[cite: 27, 28, 30]. [cite\_start]We also analyzed the distributions of RFM scores[cite: 31].
3.  [cite\_start]**Clustering Methodology**: We engineered RFM features for each customer and standardized these values to prepare the data for clustering[cite: 37, 38, 39, 40]. [cite\_start]The K-Means algorithm was chosen to segment customers into distinct groups[cite: 41]. [cite\_start]The optimal number of clusters was determined using the Elbow Method and Silhouette Score[cite: 42].
4.  [cite\_start]**Recommendation System**: An item-based collaborative filtering approach was used to build a recommendation system[cite: 49]. [cite\_start]We computed a cosine similarity matrix between products based on their co-purchase history[cite: 50]. This matrix is used to find and recommend similar products.

## 🚀 Project Deliverables

This project results in two main deliverables:

  * [cite\_start]**A Comprehensive Python Notebook**: This notebook contains the entire project workflow, from data cleaning and EDA to RFM analysis, clustering, and model evaluation[cite: 70, 71, 72, 73, 74].
  * [cite\_start]**An Interactive Streamlit Web Application**: A user-friendly application that features a product recommendation module and a customer segmentation module[cite: 75, 76, 77].

## 🛠️ Technical Stack

The project utilizes the following key libraries and technologies:

  * [cite\_start]`pandas` & `numpy` for data manipulation and analysis[cite: 68].
  * [cite\_start]`scikit-learn` for clustering (`KMeansClustering`), data preprocessing (`StandardScaler`), and similarity calculations (`CosineSimilarity`)[cite: 68].
  * `matplotlib` & `seaborn` for data visualization.
  * [cite\_start]`streamlit` for creating the web application[cite: 68].
  * `joblib` for saving and loading machine learning models.

## ▶️ How to Run the Project

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/YOUR-USERNAME/ShopperSpectrum-Project.git
    cd ShopperSpectrum-Project
    ```
2.  **Install Dependencies**:
    ```bash
    pip install pandas scikit-learn streamlit joblib numpy seaborn matplotlib
    ```
3.  **Run the Main Script**:
    First, run your main project script (e.g., `main.py` or `shopper_spectrum.py`) to generate the necessary files (`kmeans_model.pkl`, `scaler.pkl`, `product_similarity_matrix.csv`).
4.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

This will launch the interactive web application in your default browser.