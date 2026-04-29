# Deploy Solar Loss Prediction

## Files Required

Keep these files in the project root:

- `app.py`
- `ml_pipeline.py`
- `requirements.txt`
- `Procfile`
- `templates/index.html`
- `Plant_1_Generation_Data.csv.zip`
- `Plant_1_Weather_Sensor_Data.csv`

## Render Deployment

1. Create a GitHub repository.
2. Upload this project folder to the repository.
3. Go to https://render.com.
4. Create a new Web Service.
5. Connect your GitHub repository.
6. Use these settings:

   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

7. Click Deploy.
8. Render will give you a public URL.

## Railway Deployment

1. Create a GitHub repository and upload this project.
2. Go to https://railway.app.
3. Click **New Project**.
4. Choose **Deploy from GitHub repo**.
5. Select your repository.
6. Railway should detect the Python app automatically.
7. If it asks for a start command, use:

```text
gunicorn app:app
```

Railway will give you a public URL after deployment.

## Docker Deployment

This project also includes a `Dockerfile`, so it can run on Docker-based hosts.

## Local Run

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```
