# Diabetes Risk Prediction Web Application

This project is a tutorial full-stack web application that predicts whether a patient is at risk of diabetes using the `sklearn.datasets.load_diabetes` dataset.

## Features

- Frontend: HTML/CSS form for user input
- Backend: Flask REST API for prediction and data handling
- Database: SQLite storage for the diabetes dataset
- Model: Logistic regression classifier with risk labels derived from the dataset target

## Setup

1. Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Initialize the database and train the model:

```powershell
python create_db.py
```

## Run the application

```powershell
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Docker

Build the container image from the project root:

```powershell
docker build -t diabetes-risk-app .
```

Run the container:

```powershell
docker run --rm -p 5000:5000 diabetes-risk-app
```

Or run with Docker Compose:

```powershell
docker compose up --build
```

Open `http://127.0.0.1:5000` in your browser.

## Files

- `app.py` - Flask backend and API endpoints
- `templates/index.html` - frontend user interface
- `static/style.css` - styling for the web UI
- `data/diabetes.db` - SQLite database created on first run
- `requirements.txt` - Python dependencies

## API Endpoints

- `POST /api/predict` - returns diabetes risk prediction
- `GET /api/status` - returns service status and paths
