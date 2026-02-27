import streamlit as st

from app.style import render_header
from app.model_loader import load_and_train_models
from app.predict_view import render_predict_tab
from app.cluster_view import render_cluster_tab


def main():
    render_header()

    data = load_and_train_models()

    tab_predict, tab_cluster = st.tabs(
        ["Dự đoán nguy cơ trượt/đỗ", "Khám phá các nhóm sinh viên (cụm)"]
    )

    with tab_predict:
        render_predict_tab(data)

    with tab_cluster:
        render_cluster_tab(data)


if __name__ == "__main__":
    main()

