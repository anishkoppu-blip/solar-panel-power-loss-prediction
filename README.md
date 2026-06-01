☀️ Solar Panel Power Loss Prediction
A machine learning system that predicts power loss in solar panels and recommends maintenance actions — deployed as a live web application.

Live Demo: (add your Render/Railway URL here)
Dataset: Kaggle — Solar Power Generation Data


🔍 What This Project Does
Solar panels degrade over time due to soiling, temperature stress, and inverter faults. This project:

Predicts the % power loss for a solar panel given real-time weather sensor inputs (irradiation, ambient temperature, module temperature, time of day).
Classifies severity as LOW / MEDIUM / HIGH based on the predicted loss.
Recommends a maintenance action (e.g., schedule cleaning, urgent inspection).
Visualises trends through an interactive dashboard — monthly loss patterns, feature importance, loss distribution, and scatter analysis.


🏗️ Project Architecture
┌─────────────────────────────────────────────────────┐
│                   Flask Web App (app.py)             │
│  GET /           → Serves the dashboard UI          │
│  POST /predict   → Returns loss prediction + action │
│  GET /chart-data → Returns chart data for dashboard │
│  GET /health     → Model health check               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│               ML Pipeline (ml_pipeline.py)           │
│                                                      │
│  load_data()          → Kaggle CSVs or synthetic     │
│  engineer_features()  → 11 physics-informed features │
│  train_model()        → XGBoost + StandardScaler     │
│  predict()            → Single inference call        │
│  maintenance_plan()   → Rule-based action output     │
│  dashboard_chart_data()→ Chart payloads for UI       │
└──────────────────────────────────────────────────────┘

🧠 ML Pipeline Details
Features (11 total)
FeatureDescriptionHour, Day, Month, DayOfWeekTemporal features from timestampIrradiationSolar irradiance (W/m²)Irradiation_Rolling3-point rolling mean of irradianceAmbient_TemperatureOutside air temperature (°C)Module_TemperaturePanel surface temperature (°C)Temperature_DeltaModule − Ambient (thermal stress indicator)Temp_Loss_Factorγ × (T_module − T_ref) physics termDC_TheoreticalIrradiation × panel area × efficiency
Target Variable
Power_Loss_Pct — the fraction of theoretical DC power that was not generated, clipped to [0, 1].
Model
XGBoost Regressor wrapped in a sklearn Pipeline with StandardScaler.
n_estimators=220 | max_depth=4 | learning_rate=0.06
subsample=0.85   | colsample_bytree=0.85 | reg_lambda=1.5
Data Fallback
If the Kaggle CSVs are not present, the pipeline automatically generates 8,000 rows of physics-based synthetic data (sinusoidal irradiance profile + realistic noise) so the app still trains and runs end-to-end.

📊 Maintenance Decision Logic
Predicted LossSeverityAction< 5%🟢 LOWSystem optimal. Continue monitoring.5–15%🟡 MEDIUMSchedule panel cleaning within 7 days.> 15%🔴 HIGHUrgent inspection — check inverter, hot spots, bypass diodes.

🚀 Running Locally
Prerequisites

Python 3.9+
(Optional) Kaggle dataset CSVs in the project root

Steps
bash# 1. Clone the repo
git clone https://github.com/anishkoppu-blip/solar-panel-power-loss-prediction.git
cd solar-panel-power-loss-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add Kaggle data
#    Place Plant_1_Generation_Data.csv and Plant_1_Weather_Sensor_Data.csv
#    in the project root. Without them, synthetic data is used automatically.

# 4. Run the app
python app.py

# 5. Open in browser
#    http://localhost:5000
Making a Prediction (API)
bashcurl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "irradiation": 750,
    "ambient_temp": 32,
    "module_temp": 48,
    "hour": 13,
    "day": 15,
    "month": 6
  }'
Example response:
json{
  "predicted_loss_pct": 11.43,
  "severity": "MEDIUM",
  "action": "Schedule panel cleaning within 7 days. Soiling is the most likely cause.",
  "features": {
    "dc_theoretical": 209.63,
    "temperature_delta": 16.0,
    "temp_loss_factor_pct": -9.2,
    "irradiation_rolling": 750.0
  }
}

🐳 Deployment
The project ships with deployment configs for three platforms:
PlatformConfig FileCommandDockerDockerfiledocker build -t solar-app . && docker run -p 5000:5000 solar-appRenderrender.yamlPush to GitHub → auto-deployRailwayrailway.jsonrailway up

📁 Repository Structure
solar-panel-power-loss-prediction/
├── ml_pipeline.py              # Core ML logic (data → features → model → predict)
├── app.py                      # Flask API server
├── templates/
│   └── index.html              # Dashboard frontend
├── Plant_1_Generation_Data.csv.zip    # (Kaggle) DC power generation data
├── Plant_1_Weather_Sensor_Data.csv    # (Kaggle) Weather sensor data
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build
├── render.yaml                 # Render deployment config
├── railway.json                # Railway deployment config
├── Procfile                    # Gunicorn process definition
└── README_DEPLOY.md            # Extended deployment notes

🛠️ Suggested Future Improvements

 Add a Jupyter notebook (notebooks/EDA.ipynb) with exploratory data analysis and visualisations for the Kaggle dataset
 Expose model metrics (RMSE, R², MAE) on the dashboard UI
 Add cross-validation and hyperparameter tuning with GridSearchCV or Optuna
 Support multi-plant inference (Plant 2 data)
 Add unit tests with pytest for engineer_features and maintenance_plan
 Add GitHub Actions CI to run tests on every push


📦 Dependencies
flask
xgboost
scikit-learn
pandas
numpy
joblib
gunicorn

📄 License
MIT — feel free to use and adapt with attribution.
