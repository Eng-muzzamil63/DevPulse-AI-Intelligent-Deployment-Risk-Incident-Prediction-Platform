# DevPulse-AI-Intelligent-Deployment-Risk-Incident-Prediction-Platform

# DevPulse AI

Production-style ML platform for predicting the probability of a production incident before a software deployment reaches customers.

## What it solves
Engineering teams often discover deployment problems after release. DevPulse turns engineering/change metrics into a risk score, explains the strongest drivers, and provides mitigation recommendations plus a what-if simulator.

## Stack
- React + Vite + Recharts + Lucide
- FastAPI
- Scikit-learn + XGBoost
- Pandas / NumPy
- Synthetic engineering dataset
- Docker Compose

## Windows setup
### Backend
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python ml\train.py
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```
### Frontend
Open another PowerShell:
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Docker
```powershell
docker compose up --build
```

The included dataset is synthetic so the system runs locally without access to proprietary CI/CD or observability systems.
