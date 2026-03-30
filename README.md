# Smart Conseil - Cyberbullying & Harassment Detection System

## Project Overview
This project was developed during the **Integration Day at Smart Conseil**. It provides an end-to-end AI solution for detecting cyberbullying and harassment in social media text. The system transitions from experimental research (Google Colab) to a production-ready, containerized API (FastAPI + Docker).

## Features
- **Data Engineering:** Automated cleaning of noisy social media records, including duplicate removal and label standardization.
- **Database Integration:** Seamless ingestion of cleaned data into **MongoDB Atlas**.
- **Exploratory Data Analysis (EDA):** Visual analysis of class distributions and text length factors.
- **Machine Learning Pipeline:** Comparison between Probabilistic (Naive Bayes) and Geometric (Linear SVC) models.
- **Explainability:** Extraction of model feature importances to identify key linguistic factors influencing harassment.
- **Production API:** A high-performance **FastAPI** endpoint for real-time inference.
- **Containerization:** Fully Dockerized environment for scalable deployment.

---

## Technical Stack
- **Language:** Python 3.11
- **ML Libraries:** Scikit-learn, Pandas, NumPy, Joblib
- **API Framework:** FastAPI, Uvicorn
- **Database:** MongoDB Atlas (pymongo)
- **Visualization:** Matplotlib, Seaborn
- **DevOps:** Docker

---

## Project Structure
```text
ObservationDay/
├── app.py                # FastAPI Application & Inference Logic
├── Dockerfile            # Containerization instructions
├── requirements.txt      # Project dependencies
├── tfidf_vectorizer.pkl  # Trained TF-IDF Vectorizer
├── svm_model.pkl         # Trained SVM Classification Model
└── README.md             # Documentation


## 1. How to run
Launch the FastAPI server:
- **uvicorn app:app --reload
- **Test the API: Open your browser at http://localhost:8000/docs. Use the Swagger UI to send a POST request to the /predict endpoint.

### 2. Containerized Execution (Docker)
*Note: Ensure any local FastAPI servers are stopped before running Docker to avoid port conflicts.*
To run the application in a fully isolated and scalable environment:
- **Build the Docker Image:** docker build -t harassment-api.
- **Run the container:** docker run -p 8000:8000 harassment-api
- **Access the live API at:** http://localhost:8000/docs.
