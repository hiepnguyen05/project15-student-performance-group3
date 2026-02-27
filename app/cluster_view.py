import streamlit as st


def render_cluster_tab(data: dict):
    """Giao diện tab khám phá các cụm sinh viên."""
    st.subheader("Các nhóm sinh viên theo hành vi & kết quả học tập")

    st.write(
        f"Số cụm K được chọn theo Silhouette: **{data['cluster_best_k']}** "
        f"(Silhouette ≈ {data['cluster_best_silhouette']:.3f})."
    )

    cluster_ids = sorted(data["cluster_df"]["cluster"].unique())
    selected_cluster = st.selectbox(
        "Chọn cụm để xem chi tiết", options=cluster_ids, format_func=lambda x: f"Cụm {x}"
    )

    df_c = data["cluster_df"][data["cluster_df"]["cluster"] == selected_cluster]
    pass_rate = data["cluster_pass_rate"].loc[selected_cluster] * 100

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Số sinh viên trong cụm", len(df_c))
    with col_b:
        st.metric("Tỉ lệ đỗ (%)", f"{pass_rate:.1f}")
    with col_c:
        st.metric(
            "Điểm trung bình G3",
            f"{df_c['G3'].mean():.2f}",
        )

    st.markdown("#### Thống kê một số đặc trưng trong cụm")
    st.dataframe(
        data["cluster_profile"].loc[selected_cluster].to_frame().T,
        use_container_width=True,
    )

    if pass_rate < 50:
        msg = "Nhóm rủi ro cao – cần theo dõi chặt, tăng hỗ trợ học tập và giảm vắng học."
    elif pass_rate < 75:
        msg = "Nhóm rủi ro trung bình – nên có các can thiệp có mục tiêu (phụ đạo, tư vấn)."
    else:
        msg = "Nhóm rủi ro thấp – tiếp tục duy trì thói quen học tập hiện tại."

    st.info(msg)

