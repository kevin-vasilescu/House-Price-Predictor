import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import shap
import joblib

# Add src to path to import modules
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.predict import predict_from_dict
from src.models import load_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# --- Load Config and Artifacts ---
@st.cache_resource
def load_artifacts():
    """Load model, preprocessor, SHAP explainer, and other necessary data."""
    root_dir = Path(__file__).resolve().parents[1]
    model_dir = root_dir / "models"
    config_path = root_dir / "config.yaml"
    data_dir = root_dir / "data"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    try:
        preprocessor = load_model(model_dir / "preprocessor.joblib")
        model = load_model(model_dir / "xgboost.joblib") # Load the best model
    except FileNotFoundError:
        st.error(f"Model artifacts not found in `{model_dir}`. Please run `make train` first.")
        return None, None, None, None, None, None

    # Load data for visualizations
    try:
        metrics_df = pd.read_csv(model_dir / "metrics.csv", index_col=0)
        coords_df = pd.read_csv(data_dir / "neighborhood_coords.csv")
        train_df = pd.read_csv(data_dir / "raw" / "train.csv")
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e}. Please run `make setup`.")
        return None, None, None, None, None, None
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    return config, preprocessor, model, metrics_df, coords_df, train_df, explainer

config, preprocessor, model, metrics_df, coords_df, train_df, explainer = load_artifacts()

# --- Main App Logic ---
if config is None:
    st.stop() # Stop execution if artifacts failed to load

# Calculate average price by neighborhood for the map
avg_price_by_neighborhood = train_df.groupby('Neighborhood')['SalePrice'].mean().reset_index()
map_data = pd.merge(coords_df, avg_price_by_neighborhood, on='Neighborhood')


# --- UI Layout ---
st.title("🏠 Ames House Price Predictor")
st.write(
    "Welcome! This app predicts house sale prices in Ames, Iowa. "
    "Use the sidebar to input house features, or explore the model performance and data visualizations in the tabs below."
)

# --- Sidebar for User Input ---
st.sidebar.header("Enter House Features")

def user_input_features():
    """Create sidebar widgets for user input and return a dictionary."""
    feature_dict = {}
    
    # Using a mix of important features from the EDA
    feature_dict['OverallQual'] = st.sidebar.slider('Overall Quality', 1, 10, 7)
    feature_dict['GrLivArea'] = st.sidebar.number_input('Above Ground Living Area (sq ft)', 500, 5000, 1500)
    feature_dict['YearBuilt'] = st.sidebar.slider('Year Built', 1870, 2010, 2005)
    feature_dict['TotalBsmtSF'] = st.sidebar.number_input('Total Basement Area (sq ft)', 0, 6000, 1000)
    feature_dict['GarageCars'] = st.sidebar.slider('Garage Size (in cars)', 0, 5, 2)
    
    neighborhoods = sorted(train_df['Neighborhood'].unique())
    feature_dict['Neighborhood'] = st.sidebar.selectbox('Neighborhood', neighborhoods, index=neighborhoods.index('CollgCr'))
    
    kitchen_qual_map = config['features']['categorical']['ordinal_mappings']['KitchenQual']
    feature_dict['KitchenQual'] = st.sidebar.select_slider('Kitchen Quality', options=kitchen_qual_map, value='Gd')
    
    # Add other necessary columns with default values so the model doesn't break
    # This is a simplification; a real app might need more comprehensive inputs.
    all_model_features = preprocessor.get_feature_names_out()
    all_raw_features = list(config['features']['numerical']) + \
                       list(config['features']['categorical']['nominal']) + \
                       list(config['features']['categorical']['ordinal'].keys())
                       
    for feat in all_raw_features:
        if feat not in feature_dict:
            # Use median for numeric, mode for categorical as a default
            if train_df[feat].dtype in ['int64', 'float64']:
                feature_dict[feat] = train_df[feat].median()
            else:
                feature_dict[feat] = train_df[feat].mode()[0]
    
    # Use some inputs for required engineered features
    feature_dict['YrSold'] = 2025 # Assume prediction for current year
    feature_dict['YearRemodAdd'] = feature_dict['YearBuilt']
    feature_dict['FullBath'] = 2
    feature_dict['HalfBath'] = 0
    feature_dict['BsmtFullBath'] = 0
    feature_dict['BsmtHalfBath'] = 0
    feature_dict['1stFlrSF'] = feature_dict['TotalBsmtSF']
    feature_dict['2ndFlrSF'] = feature_dict['GrLivArea'] - feature_dict['1stFlrSF']
    feature_dict['GarageYrBlt'] = feature_dict['YearBuilt']

    return feature_dict

input_dict = user_input_features()


# --- Main Panel: Prediction and Explanations ---
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Prediction Result")
    if st.sidebar.button("Predict Price", use_container_width=True, type="primary"):
        with st.spinner("Calculating..."):
            # Predict
            prediction = predict_from_dict(input_dict, Path("models"))
            
            st.metric(label="Predicted Sale Price", value=f"${prediction:,.2f}")

            # Explain Prediction
            input_df = pd.DataFrame([input_dict])
            X_processed = build_features(input_df, fit_mode=False, preprocessor=preprocessor)
            
            shap_values = explainer.shap_values(X_processed)
            
            # Create a SHAP force plot using Plotly
            feature_names = preprocessor.get_feature_names_out()
            base_value = explainer.expected_value
            
            st.write("---")
            st.subheader("Prediction Explanation")
            st.write("What pushed the price up or down? (based on SHAP values)")

            # Simple text explanation
            shap_df = pd.DataFrame({
                'feature': feature_names,
                'shap_value': shap_values[0]
            })
            shap_df['abs_shap'] = shap_df['shap_value'].abs()
            top_features = shap_df.sort_values('abs_shap', ascending=False).head(5)
            
            for _, row in top_features.iterrows():
                if row['shap_value'] > 0:
                    st.success(f"**{row['feature']}** increased the prediction.")
                else:
                    st.error(f"**{row['feature']}** decreased the prediction.")
    else:
        st.info("Click 'Predict Price' in the sidebar to see a result.")

with col2:
    st.subheader("SHAP Feature Importance")
    st.write("This plot shows the impact of each feature on the model's output for the given input.")
    
    try:
        if 'shap_values' in locals():
            # Generate a bar plot for the single prediction
            shap_df_display = shap_df.copy()
            shap_df_display['positive'] = shap_df_display['shap_value'] > 0
            
            fig = px.bar(
                top_features.sort_values('shap_value'),
                x='shap_value', 
                y='feature',
                color='positive',
                color_discrete_map={True: 'green', False: 'red'},
                orientation='h',
                labels={'shap_value': 'SHAP Value (impact on log-price)', 'feature': 'Feature'},
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    except NameError:
        st.write("A SHAP plot will appear here after a prediction is made.")

# --- Tabs for Additional Info ---
tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "🗺️ Neighborhood Price Map", "ℹ️ Feature Info"])

with tab1:
    st.subheader("Model Performance Comparison")
    st.write("Here's how the different models performed during cross-validation (lower is better).")
    if not metrics_df.empty:
        fig = px.bar(
            metrics_df, 
            y='CV_RMSE_log', 
            x=metrics_df.index,
            title="Cross-Validated Root Mean Squared Error (on Log-Transformed Price)",
            labels={'index': 'Model', 'CV_RMSE_log': 'CV RMSE (log scale)'},
            text_auto='.4f'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Metrics file `models/metrics.csv` not found.")

with tab2:
    st.subheader("Average Sale Price by Neighborhood")
    st.warning("Note: The map coordinates are placeholders and do not represent the true geographic locations in Ames, IA.")
    
    if not map_data.empty:
        fig = px.scatter_mapbox(
            map_data,
            lat="Latitude",
            lon="Longitude",
            size="SalePrice",
            color="SalePrice",
            hover_name="Neighborhood",
            hover_data={"SalePrice": ":$,.0f", "Latitude": False, "Longitude": False},
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=30,
            zoom=11,
            mapbox_style="carto-positron",
            title="Average House Price by Neighborhood"
        )
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Feature Definitions")
    st.markdown("""
    - **Overall Quality**: Rates the overall material and finish of the house (1-10).
    - **Above Ground Living Area**: Total square feet of living space above ground.
    - **Year Built**: Original construction date.
    - **Total Basement Area**: Total square feet of basement area.
    - **Garage Size**: Size of garage in car capacity.
    - **Neighborhood**: Physical location within Ames city limits.
    - **Kitchen Quality**: Quality of the kitchen (Excellent, Good, Typical, Fair, Poor).
    """)