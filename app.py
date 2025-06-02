import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Import SimpleImputer
import numpy as np

# --- Configuration and Caching ---

st.set_page_config(
    page_title="Multi-Format Data Analyzer & ML Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data(file_buffer, file_type):
    """
    Loads data from a file buffer into a Pandas DataFrame based on file type.
    Caches the result for performance.
    Applies basic numeric coercion to handle mixed data types in columns.
    """
    df = None
    try:
        if file_type == "csv":
            df = pd.read_csv(file_buffer)
        elif file_type == "xlsx" or file_type == "xls":
            df = pd.read_excel(file_buffer)
        elif file_type == "json":
            df = pd.read_json(file_buffer)
        elif file_type == "parquet":
            df = pd.read_parquet(file_buffer)
        elif file_type == "tsv":
            df = pd.read_csv(file_buffer, sep='\t')
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None

        # Attempt to convert all columns to numeric, coercing errors to NaN.
        # This handles mixed data types gracefully, converting non-numeric entries to NaN.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # If the column is now float and contains only integers, convert to nullable integer type.
            if pd.api.types.is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all():
                df[col] = df[col].astype('Int64') # Pandas nullable integer type

        return df
    except Exception as e:
        st.error(f"An error occurred during file loading or initial type conversion: {e}")
        return None

# --- UI Functions ---

def display_dataframe_summary(df):
    """Displays a summary of the DataFrame including head, columns, types, and description."""
    st.subheader("Data Overview")

    with st.expander("View Raw Data (First 5 Rows)"):
        st.dataframe(df.head(), use_container_width=True) # Removed use_table_with_columns=False

    with st.expander("Column Names and Data Types"):
        st.write("### Column Names:")
        st.write(df.columns.tolist())
        st.write("### Data Types:")
        st.write(df.dtypes)

    with st.expander("Basic Statistical Summary"):
        st.write(df.describe())

    with st.expander("Missing Values Count"):
        st.write(df.isnull().sum())

def perform_column_analysis(df, selected_column):
    """Performs and displays analysis for a selected column, including basic stats and visualizations."""
    st.subheader(f"Analysis for Column: '{selected_column}'")

    if pd.api.types.is_numeric_dtype(df[selected_column]):
        st.write("### Numeric Column Statistics:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Mean", value=f"{df[selected_column].mean():.2f}")
            st.metric(label="Median", value=f"{df[selected_column].median():.2f}")
        with col2:
            st.metric(label="Standard Deviation", value=f"{df[selected_column].std():.2f}")
            st.metric(label="Min Value", value=f"{df[selected_column].min():.2f}")
        with col3:
            st.metric(label="Max Value", value=f"{df[selected_column].max():.2f}")
            st.metric(label="Count (non-null)", value=f"{df[selected_column].count()}")

        st.write("### Distribution Plot:")
        fig = px.histogram(df, x=selected_column, marginal="box",
                           title=f"Distribution of {selected_column}",
                           template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    else: # Categorical or Object type column
        st.write("### Categorical Column Details:")
        st.write(f"**Number of Unique Values:** {df[selected_column].nunique()}")
        st.write("**Unique Values and Their Counts:**")
        value_counts_df = df[selected_column].value_counts().reset_index()
        value_counts_df.columns = ['Value', 'Count']
        st.dataframe(value_counts_df, use_container_width=True) # Removed use_table_with_columns=False

        st.write("### Value Counts Bar Chart:")
        fig = px.bar(value_counts_df, x='Value', y='Count',
                     title=f"Value Counts for {selected_column}",
                     template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

def perform_ml_classification(df):
    st.subheader("Machine Learning: Classification")
    st.markdown("Here, you can train a classification model to predict a categorical outcome (like 'Heart Attack Risk') based on other features in your data.")

    # 1. Select Target Variable
    target_column = st.selectbox(
        "Select the target column (the column you want to predict):",
        options=df.columns.tolist(),
        index=df.columns.get_loc('output') if 'output' in df.columns else 0 # Default to 'output' if exists
    )

    # Filter out the target column from features
    feature_columns = [col for col in df.columns if col != target_column]

    # 2. Select Features (Predictors)
    selected_features = st.multiselect(
        "Select feature columns (columns to use for prediction):",
        options=feature_columns,
        default=feature_columns # Select all by default
    )

    if not selected_features:
        st.warning("Please select at least one feature column to train the model.")
        return

    st.markdown("---")
    st.write("### Model Training Parameters")

    # 3. Handle Missing Values (Simple Imputation)
    st.info("Missing values in numeric features will be filled with the mean. Categorical features will be handled by One-Hot Encoding.")
    
    # Separate numeric and categorical features among selected features
    numeric_features = df[selected_features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[selected_features].select_dtypes(include='object').columns.tolist()

    # Preprocessing pipelines for numerical and categorical features
    # Numerical: Impute missing values with mean
    # Categorical: One-hot encode
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')), # Use SimpleImputer
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # Define X and y
    X = df[selected_features]
    y = df[target_column]

    # Ensure target is suitable for classification (e.g., binary or discrete)
    # If target is float and all values are integers, convert to int
    if pd.api.types.is_float_dtype(y) and all(y.dropna() == y.dropna().astype(int)):
        y = y.astype(int)
        st.info(f"Target column '{target_column}' converted from float to int.")

    if not (pd.api.types.is_numeric_dtype(y) and y.nunique() <= 20): # Heuristic for classification target
        st.warning(f"Selected target column '{target_column}' may not be suitable for classification (too many unique values or non-numeric). Please ensure it's a discrete categorical variable.")
        try:
            # Only attempt conversion if it's not already numeric
            if not pd.api.types.is_numeric_dtype(y):
                y = y.astype('category').cat.codes
                st.info(f"Target column '{target_column}' converted to numerical categories for classification.")
        except Exception as e:
            st.error(f"Cannot convert target column to numerical categories: {e}. Please choose a suitable target.")
            return

    # 4. Split Data
    test_size = st.slider("Test Set Size (e.g., 0.2 for 20%)", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State (for reproducibility)", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    st.write(f"Training data size: {X_train.shape[0]} rows")
    st.write(f"Test data size: {X_test.shape[0]} rows")

    # 5. Train Model (Logistic Regression)
    st.markdown("#### Model Selection: Logistic Regression")
    
    # Create the model pipeline
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(random_state=random_state, solver='liblinear'))]) # liblinear for small datasets and binary classification

    if st.button("Train Classification Model"):
        with st.spinner("Training model... This might take a moment."):
            try:
                model.fit(X_train, y_train)
                st.success("Model trained successfully!")

                # 6. Evaluate Model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.write("### Model Evaluation")
                st.metric(label="Accuracy Score", value=f"{accuracy:.2f}")

                st.write("#### Classification Report")
                st.text(classification_report(y_test, y_pred))

                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            xticklabels=model.named_steps['classifier'].classes_, yticklabels=model.named_steps['classifier'].classes_)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                ax_cm.set_title('Confusion Matrix')
                st.pyplot(fig_cm)

                # 7. Insights for "Groups More Prone"
                st.write("### Insights: Which groups are more prone to heart attack?")
                st.info("For Logistic Regression, the coefficients can indicate the influence of features. Positive coefficients suggest an increased likelihood of the predicted class (e.g., heart attack risk=1), while negative coefficients suggest a decreased likelihood.")

                # Get feature names after one-hot encoding
                try:
                    preprocessor_fitted = model.named_steps['preprocessor']
                    feature_names_out = preprocessor_fitted.get_feature_names_out()
                    
                    if hasattr(model.named_steps['classifier'], 'coef_'):
                        coefficients = model.named_steps['classifier'].coef_[0]
                        
                        coef_df = pd.DataFrame({'Feature': feature_names_out, 'Coefficient': coefficients})
                        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
                        st.write("#### Feature Coefficients (Influence on Prediction)")
                        st.dataframe(coef_df)
                        st.write("Higher positive coefficients indicate a stronger positive relationship with the predicted class (e.g., heart attack risk).")
                        st.write("Higher negative coefficients indicate a stronger negative relationship with the predicted class.")

                    else:
                        st.write("Coefficients are not directly available for this model type.")

                except Exception as e:
                    st.error(f"Error retrieving feature coefficients: {e}")

            except Exception as e:
                st.error(f"An error occurred during model training or evaluation: {e}")

def perform_advanced_visualizations(df):
    st.subheader("Advanced Visualizations")

    st.markdown("### Correlation Heatmap")
    st.info("A correlation heatmap shows the correlation coefficients between all pairs of numerical variables. Values closer to 1 or -1 indicate strong positive or negative correlations, respectively. Values close to 0 indicate weak or no linear correlation. This can help identify features that are strongly related to each other or to your target variable.")

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        st.warning("No numeric columns found for correlation heatmap.")
        return

    corr = numeric_df.corr()

    # Create the heatmap using matplotlib and seaborn
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
    plt.title('Correlation Heatmap of Numerical Features')
    st.pyplot(fig)

    st.markdown("---") # Separator

    st.markdown("### Feature-Target Relationship Visualizations")
    st.info("These plots help visualize the relationship between each feature and the selected target variable. This can provide insights into how different feature values correlate with the outcome.")

    # Select target column for visualization
    target_column_for_viz = st.selectbox(
        "Select the target column to visualize relationships with other features:",
        options=df.columns.tolist(),
        index=df.columns.get_loc('output') if 'output' in df.columns else 0,
        key="target_viz_select"
    )

    if not target_column_for_viz:
        st.warning("Please select a target column to visualize relationships.")
        return

    # List features excluding the target
    features_for_viz = [col for col in df.columns if col != target_column_for_viz]

    if not features_for_viz:
        st.info("No other features available to visualize relationships with the target.")
        return

    # Loop through each feature and plot
    for feature in features_for_viz:
        st.write(f"#### Relationship between '{feature}' and '{target_column_for_viz}'")

        if pd.api.types.is_numeric_dtype(df[feature]):
            # Box Plot or Violin Plot for numeric features
            fig = px.box(df, x=target_column_for_viz, y=feature,
                         title=f"{feature} vs. {target_column_for_viz}",
                         labels={target_column_for_viz: f"{target_column_for_viz} (0: No Risk, 1: Risk)", feature: feature},
                         template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Interpretation:** This box plot shows the distribution of '{feature}' across the '{target_column_for_viz}' classes (e.g., No Risk and Risk). The line inside the box represents the median, the box edges represent quartiles, and whiskers represent the typical data range. Dots are outliers.")
        else:
            # Bar Plot for categorical features
            grouped_counts = df.groupby([feature, target_column_for_viz]).size().reset_index(name='Count')
            
            fig = px.bar(grouped_counts, x=feature, y='Count', color=target_column_for_viz,
                         title=f"Distribution of {feature} by {target_column_for_viz}",
                         labels={target_column_for_viz: f"{target_column_for_viz} (0: No Risk, 1: Risk)"},
                         barmode='group',
                         template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Interpretation:** This bar chart displays the distribution of '{target_column_for_viz}' classes for each category within '{feature}'. The height of the bars represents the count of each combination in the dataset. For example, you can see the distribution of males and females in the 'sex' column across heart attack risk (0 or 1).")
        st.markdown("---") # Separator after each plot

# --- Main Application Logic ---

def main():
    st.title("📊 Multi-Format Data Analyzer & ML Predictor")
    st.markdown("Upload your data in various formats (CSV, Excel, JSON, Parquet, TSV) to get comprehensive statistical insights, advanced visualizations, and even train machine learning models.")

    # Abbreviations explanation section
    with st.expander("Understand Data Set Abbreviations"):
        st.markdown("""
        This application is designed to analyze medical datasets, especially those related to heart disease. Below are the meanings of common abbreviations you might encounter in the datasets you upload:

        * **`age`**: Patient's age (in years).
        * **`sex`**: Sex (0 = Female, 1 = Male).
        * **`cp`**: Chest Pain Type (categorical values 0-3; indicates different chest pain syndromes).
            * `0`: Typical angina
            * `1`: Atypical angina
            * `2`: Non-anginal pain
            * `3`: Asymptomatic
        * **`trtbps`**: Resting Blood Pressure (in mm Hg).
        * **`chol`**: Serum Cholesterol (in mg/dl).
        * **`fbs`**: Fasting Blood Sugar (> 120 mg/dl is 1, else 0). An indicator for diabetes.
        * **`restecg`**: Resting Electrocardiographic (ECG) Results (categorical values 0-2; indicates abnormalities in heart's electrical activity).
            * `0`: Normal
            * `1`: ST-T wave abnormality
            * `2`: Probable or definite left ventricular hypertrophy
        * **`thalachh`**: Maximum Heart Rate Achieved (in bpm).
        * **`exng`**: Exercise Induced Angina (1 = Yes, 0 = No). Chest pain experienced during exercise.
        * **`oldpeak`**: ST Depression Induced by Exercise Relative to Rest (amount of ST segment depression on ECG during exercise).
        * **`slp`**: Slope of the Peak Exercise ST Segment (categorical values 0-2; characteristic of the ST segment slope during exercise test).
            * `0`: Up-sloping
            * `1`: Flat
            * `2`: Down-sloping
        * **`caa`**: Number of Major Vessels Colored by Fluoroscopy (0-3; indicates severity of coronary artery disease).
        * **`thall`**: Thalassemia (Thallium Stress Test Result) (categorical values 1-3; assessment of blood flow to heart muscle).
            * `1`: Normal
            * `2`: Fixed defect
            * `3`: Reversible defect
        * **`output`**: Target Variable (0 = No heart attack risk, 1 = Heart attack risk).

        Understanding these abbreviations will help you interpret the patterns in your dataset and the predictions made by the machine learning model more accurately.
        """)

    with st.sidebar:
        st.header("Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls", "json", "parquet", "tsv"]
        )
        st.markdown("---")
        st.info("Your data is processed in-memory and not stored.")

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        df = load_data(uploaded_file, file_extension)

        if df is not None:
            st.success(f"File '{uploaded_file.name}' successfully loaded! Analyzing your data...")

            tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Column Analysis", "ML Classification", "Advanced Visualizations"])

            with tab1:
                display_dataframe_summary(df)

            with tab2:
                st.subheader("Individual Column Analysis")
                selected_column = st.selectbox(
                    "Select a column to perform detailed analysis and visualization:",
                    df.columns,
                    key="column_analysis_select"
                )
                if selected_column:
                    perform_column_analysis(df, selected_column)
                else:
                    st.info("No column selected for detailed analysis.")

            with tab3:
                perform_ml_classification(df.copy())

            with tab4:
                perform_advanced_visualizations(df)

        else:
            st.info("Please upload a valid data file.")
    else:
        st.info("Please upload a data file from the sidebar to begin your analysis.")

if __name__ == "__main__":
    main()