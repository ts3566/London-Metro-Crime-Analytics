# London Burglary Analysis and Prediction

This project analyzes residential burglary patterns across London wards and LSOAs using Metropolitan Police data. It provides XGBoost-based predictions and interactive visualizations to aid police resource allocation.

## Raw Data Source and Storage

**Data Source**: Download monthly street-level crime data from [data.police.uk](https://data.police.uk/data/)
- Select "Metropolitan Police" for the force
- Download monthly CSV files

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   ```bash
   python scripts/ingest_burglary.py          # ~3 minutes
   python scripts/xgboost_predictions_12m.py  # ~10 minutes  
   streamlit run scripts/streamlit_app_12m.py # ~30 seconds
   ```

## XGBoost Features
The model uses the following features for burglary prediction:

**Time Components**:
- `year`, `month`, `quarter` - Temporal identifiers
- `is_summer`, `is_winter` - Seasonal indicators

**Historical Data (Lags)**:
- `burglaries_lag_1/2/3/6/12` - Previous month burglary counts

**Rolling Statistics**:
- `burglaries_rollmean_3/6/12` - Rolling averages
- `burglaries_rollstd_3/6/12` - Rolling standard deviations

**Socio-economic Indicators**:
- `population` - Ward population
- `imd_score` - Index of Multiple Deprivation
- `income_score`, `employment_score`, `crime_score` - Deprivation measures
- `health_score`, `housing_score`, `environment_score` - Quality of life indicators
