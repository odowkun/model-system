import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import os
from fastapi import HTTPException
from dto import ContentRequest, CollaborativeRequest, HybridRequest
from utils import load_data_from_url, load_model_from_url, load_technicians_df
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
technicians_df_url = "https://storage.googleapis.com/model-recommend/technicians.csv"

# Load models and preprocessing objects
try:
    collaborative_model_url = "https://storage.googleapis.com/model-recommend/models/collaborative_filtering.h5"
    content_based_model_url = "https://storage.googleapis.com/model-recommend/models/content_based_filtering.h5"

    collaborative_model_data = load_model_from_url(collaborative_model_url)
    content_based_model_data = load_model_from_url(content_based_model_url)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_collab_file:
        tmp_collab_path = tmp_collab_file.name
        tmp_collab_file.write(collaborative_model_data)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp_content_file:
        tmp_content_path = tmp_content_file.name
        tmp_content_file.write(content_based_model_data)

    collaborative_model = tf.keras.models.load_model(tmp_collab_path)
    content_based_model = tf.keras.models.load_model(tmp_content_path)

    os.remove(tmp_collab_path)
    os.remove(tmp_content_path)

except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")

try:
    tfidf = load_data_from_url('https://storage.googleapis.com/model-recommend/models/tfidf_vectorizer.pkl', is_pickle=True)
    scaler_experience = load_data_from_url('https://storage.googleapis.com/model-recommend/models/scaler_experience.pkl', is_pickle=True)
    scaler_ratings = load_data_from_url('https://storage.googleapis.com/model-recommend/models/scaler_ratings.pkl', is_pickle=True)
    encoder = load_data_from_url('https://storage.googleapis.com/model-recommend/models/encoder.pkl', is_pickle=True)
    user_id_map = load_data_from_url('https://storage.googleapis.com/model-recommend/models/user_id_map.pkl', is_pickle=True)
    technician_id_map = load_data_from_url('https://storage.googleapis.com/model-recommend/models/technician_id_map.pkl', is_pickle=True)
except Exception as e:
    raise RuntimeError(f"Error loading preprocessing objects: {e}")

# Load technicians data
technicians_df = load_technicians_df(technicians_df_url)
ori_technician = technicians_df.copy()

# Preprocessing
tfidf_vectorizer = tfidf
skills_tfidf = tfidf_vectorizer.transform(technicians_df['skills']).toarray()

# Scale experience and ratings
technicians_df['experience'] = scaler_experience.transform(technicians_df[['experience']])
technicians_df['ratingsreceived'] = scaler_ratings.transform(technicians_df[['ratingsreceived']])

# Encode certifications
certifications_encoded_sparse = encoder.transform(technicians_df[['certifications']])
certifications_encoded = certifications_encoded_sparse.toarray()

X_exp = technicians_df['experience'].values.reshape(-1, 1)
X_rating = technicians_df['ratingsreceived'].values.reshape(-1, 1)
X_cert = certifications_encoded

# Combine features into a single array
X = np.hstack([skills_tfidf, X_exp, X_cert, X_rating])

def content_based_filtering(request: ContentRequest):
    user_skill = request.user_skill

    try:
        user_skill_tfidf = tfidf_vectorizer.transform([user_skill]).toarray()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing user skill: {e}")

    X_input = np.hstack([user_skill_tfidf, np.zeros((1, X.shape[1] - user_skill_tfidf.shape[1]))])

    try:
        predicted_score = content_based_model.predict(X_input).flatten()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting score: {e}")

    best_match_score = -1
    best_technician_index = -1

    for idx in range(X.shape[0]):
        technician = technicians_df.iloc[idx]
        skill_match = user_skill.lower() in technician['skills'].lower()
        if skill_match:
            combined_score = (predicted_score + technician['experience'] +
                              technician['ratingsreceived'] +
                              certifications_encoded_sparse[idx].sum())
            if combined_score > best_match_score:
                best_match_score = combined_score
                best_technician_index = idx

    if best_technician_index != -1:
        best_technician = ori_technician.iloc[best_technician_index]
        # Convert any numpy types to native Python types
        best_technician = best_technician.apply(lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x)
        return {"message": best_technician.to_dict()}
    else:
        raise HTTPException(status_code=404, detail="No matching technician found.")

def collaborative_filtering(request: CollaborativeRequest):
    user_id = request.user_id

    if user_id not in user_id_map:
        raise HTTPException(status_code=404, detail="User ID not found.")

    user_idx = user_id_map[user_id]

    try:
        user_input = np.array([user_idx] * len(technician_id_map))
        technician_input = np.arange(len(technician_id_map))
        predicted_ratings = collaborative_model.predict([user_input, technician_input])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting ratings: {e}")

    predictions = [(tech_id, pred) for tech_id, pred in zip(technician_id_map.keys(), predicted_ratings.flatten())]
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    top_recommendation = sorted_predictions[0]
    top_technician = ori_technician[technicians_df['technicianid'] == top_recommendation[0]].iloc[0]

    # Convert any numpy types to native Python types
    top_technician = top_technician.apply(lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x)
    return {"message": top_technician.to_dict()}

def hybrid_recommendation(request: HybridRequest):
    user_id = request.user_id
    user_skill = request.user_skill

    # Content-based part
    try:
        user_skill_tfidf = tfidf_vectorizer.transform([user_skill]).toarray()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing user skill: {e}")

    X_content_input = np.hstack([user_skill_tfidf, np.zeros((1, X.shape[1] - user_skill_tfidf.shape[1]))])

    try:
        predicted_content_score = content_based_model.predict(X_content_input).flatten()[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting content-based score: {e}")

    # Collaborative part
    if user_id not in user_id_map:
        raise HTTPException(status_code=404, detail="User ID not found.")

    user_idx = user_id_map[user_id]

    try:
        user_input = np.array([user_idx] * len(technician_id_map))
        technician_input = np.arange(len(technician_id_map))
        predicted_ratings = collaborative_model.predict([user_input, technician_input])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting collaborative ratings: {e}")

    # Hybrid recommendation
    predictions = [(tech_id, (pred + predicted_content_score) / 2) for tech_id, pred in zip(technician_id_map.keys(), predicted_ratings.flatten())]
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    top_recommendation = sorted_predictions[0]
    top_technician = ori_technician[technicians_df['technicianid'] == top_recommendation[0]].iloc[0]

    # Convert any numpy types to native Python types
    top_technician = top_technician.apply(lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x)
    return {"message": top_technician.to_dict()}
