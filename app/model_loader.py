import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).parent.parent

# Đảm bảo có thể import src nếu cần mở rộng sau này
sys.path.append(str(PROJECT_ROOT))


@st.cache_resource
def load_and_train_models():
    """
    Load dữ liệu gốc, huấn luyện mô hình phân lớp + phân cụm.
    Hàm này được cache để không phải train lại mỗi lần người dùng nhập form.
    """
    from src.data.loader import DataLoader

    loader = DataLoader()
    df = loader.load_combined_data(merge=False)

    # Tạo biến mục tiêu pass/fail từ G3
    df["pass_fail"] = (df["G3"] >= 10).astype(int)

    # Đặc trưng mô hình hóa (giống notebook 05)
    feature_columns = [
        "school",
        "sex",
        "age",
        "address",
        "famsize",
        "Pstatus",
        "Medu",
        "Fedu",
        "Mjob",
        "Fjob",
        "reason",
        "guardian",
        "traveltime",
        "studytime",
        "failures",
        "schoolsup",
        "famsup",
        "paid",
        "activities",
        "nursery",
        "higher",
        "internet",
        "romantic",
        "famrel",
        "freetime",
        "goout",
        "Dalc",
        "Walc",
        "health",
        "absences",
        "G1",
        "G2",
    ]

    # Mã hóa biến phân loại
    df_encoded = df.copy()
    label_encoders: dict[str, LabelEncoder] = {}

    for col in feature_columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    X = df_encoded[feature_columns].values
    y = df_encoded["pass_fail"].values

    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Mô hình phân lớp chính: RandomForest
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_scaled, y)

    # Chuẩn bị cho phân cụm
    cluster_features = [
        "age",
        "Medu",
        "Fedu",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2",
        "G3",
    ]
    X_cluster = df[cluster_features].copy()
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

    # Tìm K tốt nhất dựa trên silhouette trong khoảng 2–6
    best_k = 2
    best_sil = -1.0
    for k in range(2, 7):
        kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = kmeans_tmp.fit_predict(X_cluster_scaled)
        sil = silhouette_score(X_cluster_scaled, labels_tmp)
        if sil > best_sil:
            best_sil = sil
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_cluster_scaled)
    df_clusters = df.copy()
    df_clusters["cluster"] = cluster_labels

    # Profile cụm
    cluster_profile = (
        df_clusters.groupby("cluster")[
            ["G3", "studytime", "failures", "absences", "goout", "Dalc", "Walc"]
        ]
        .agg(["mean", "count"])
        .round(2)
    )

    pass_rate = (
        df_clusters.groupby("cluster")["pass_fail"].mean().rename("pass_rate")
    )

    return {
        "raw_df": df,
        "feature_columns": feature_columns,
        "label_encoders": label_encoders,
        "scaler": scaler,
        "clf": clf,
        "X_scaled": X_scaled,
        "y": y,
        "cluster_features": cluster_features,
        "cluster_scaler": scaler_cluster,
        "cluster_model": kmeans,
        "cluster_df": df_clusters,
        "cluster_profile": cluster_profile,
        "cluster_pass_rate": pass_rate,
        "cluster_best_k": best_k,
        "cluster_best_silhouette": best_sil,
    }


def encode_single_input(form_data: dict, feature_columns, label_encoders, scaler):
    """Chuyển dữ liệu từ form Streamlit thành vector đặc trưng đã mã hóa + scale."""
    row = {col: form_data[col] for col in feature_columns}
    df_input = pd.DataFrame([row])

    for col, le in label_encoders.items():
        df_input[col] = le.transform(df_input[col].astype(str))

    X = df_input[feature_columns].values
    X_scaled = scaler.transform(X)
    return X_scaled

