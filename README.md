# 📊 Heart Disease Data Analyzer & ML Predictor

This project is a Streamlit web application designed for comprehensive data analysis and machine learning classification, specifically focusing on heart disease datasets. It allows users to upload various data formats (CSV, Excel, JSON, Parquet, TSV) and provides immediate insights, visualizations, and a trained Logistic Regression model. A sample `heart.csv` dataset is included in the repository for quick demonstration and testing.

## ✨ Features

* **Multi-Format Data Upload:** Supports CSV, Excel (.xlsx, .xls), JSON, Parquet, and TSV files.
* **Sample Data Included:** Comes with a pre-loaded `heart.csv` dataset for quick demonstration. This allows users to test the application immediately without needing to upload a file.
* **Data Overview:** Displays raw data (first 5 rows), column names, data types, basic statistical summary, and missing value counts.
* **Individual Column Analysis:** Provides detailed statistics and visualizations (histograms for numeric, bar charts for categorical) for any selected column.
* **Machine Learning Classification:**
    * Trains a Logistic Regression model.
    * Allows selection of target and feature columns.
    * Handles missing values using mean imputation for numeric features and One-Hot Encoding for categorical features.
    * Provides model evaluation metrics (Accuracy, Classification Report, Confusion Matrix).
    * Shows feature coefficients to explain variable influence on prediction.
* **Advanced Visualizations:**
    * Interactive Correlation Heatmap for numerical features to identify relationships.
    * Feature-Target Relationship Visualizations (Box plots for numeric, Bar charts for categorical features vs. target) to explore how different variables relate to the outcome.
* **Abbreviations Explanation:** An in-app collapsible section provides definitions for common medical abbreviations found in heart disease datasets, aiding in data interpretation.

## 🚀 How to Run the Application Locally

To run this application on your local machine, follow these steps:

1.  **Clone the Repository:**
    First, clone this GitHub repository to your local machine:
    ```bash
    git clone [https://github.com/CosmicMass/heart-disease-data-analyzer-ml.git](https://github.com/CosmicMass/heart-disease-data-analyzer-ml.git)
    cd heart-disease-data-analyzer-ml
    ```

2.  **Create a Virtual Environment (Recommended):**
    It's good practice to use a virtual environment to manage project dependencies:
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    Install all required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit Application:**
    Once all dependencies are installed, you can start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

6.  **Access the Application:**
    The application will automatically open in your default web browser (typically at `http://localhost:8501`).
    You can either upload your own data files or, for immediate testing, click the **"Load Sample Heart.csv"** button in the sidebar. The `heart.csv` file is included in this repository, making it easy to get started right away.

## 📄 License

This project is licensed under the [MIT License](LICENSE).