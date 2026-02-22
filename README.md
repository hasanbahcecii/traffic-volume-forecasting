# Traffic Volume Forecasting with GRU (Gated Recurrent Unit)

This project is a **Python-based deep learning system** designed to forecast traffic volume using historical traffic data. A GRU (Gated Recurrent Unit) model is trained to predict future traffic flow, enabling smarter urban planning and traffic management.

---

## 📌 Problem Definition

The goal is to predict future traffic volume based on historical traffic data. Accurate forecasting of traffic flow is crucial for several applications, including:

- Urban planning

- Traffic light optimization

- Reducing congestion

- Improving emergency response times

By anticipating traffic volume, transportation systems can be better managed, leading to improved infrastructure efficiency and smoother mobility across cities.

---

## 📊 Dataset

[Metro Interstate Traffic Volume (UCI)](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume)

- Time period: 2012–2018

Hourly measurements

- ~48,000 samples

- Target variable: traffic_volume

- Features:

    date_time (timestamp)

    holiday (official holiday indicator)

    temp (temperature in Kelvin)

    rain and snow

    clouds_all (cloud coverage)

    weather_main (weather condition)

---

## 🛠️ Technologies

- PyTorch → GRU-based time series forecasting model

- FastAPI → Serve the trained model as a REST API

- Streamlit → Build a web-based user interface

---

## 📂 Project Structure
```Code

traffic-volume-forecasting/
│
├── load_and_explore.py     # Data analysis and exploration
├── preprocessing.py        # Data preprocessing
├── train.py                # Model training
├── test.py                 # Model testing and evaluation
├── main_api.py             # FastAPI service
├── test_requests.py        # FastAPI request testing
├── app_streamlit.py        # Streamlit web app
├── requirements.txt        # Dependency list
└── .gitignore              # Excludes unnecessary files
```

---

## ⚙️ Setup

Clone the repository:

```bash

git clone git@github.com:hasanbahcecii/traffic-volume-forecasting.git
cd traffic-volume-forecasting
```
Create and activate a virtual environment:

```bash

python3 -m venv venv
source venv/bin/activate
```
Install dependencies:

```bash

pip install -r requirements.txt
```

---

## 🚀 Usage

Data Analysis

```bash

python load_and_explore.py
```
Preprocessing

```bash

python preprocessing.py
```
Model Training

```bash

python train.py
```
Model Testing

```bash

python test.py
```
Run FastAPI Service

```bash

uvicorn main_api:app --reload
```
Test FastAPI Requests

```bash

python test_requests.py
```
Launch Streamlit App

```bash

streamlit run app_streamlit.py
```

---

## 📊 Outputs

- Training and validation loss/accuracy plots

- Trained GRU model saved for deployment

- REST API for real-time predictions

- Streamlit dashboard for interactive visualization

---

## 📜 License

MIT