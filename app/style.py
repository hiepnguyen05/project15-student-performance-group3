import streamlit as st


def render_header():
    """Thiết lập cấu hình trang và style chung cho ứng dụng."""
    st.set_page_config(
        page_title="Hệ thống hỗ trợ học tập sinh viên",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main-title {
            font-size: 30px;
            font-weight: 700;
            color: #1f4e79;
            margin-bottom: 4px;
        }
        .subtitle {
            font-size: 16px;
            color: #555;
            margin-bottom: 20px;
        }
        .section-card {
            background-color: #f9fafb;
            padding: 18px 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            margin-bottom: 12px;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 12px 14px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        .risk-box {
            padding: 12px 14px;
            border-radius: 8px;
            margin-top: 10px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='main-title'>Hệ thống hỗ trợ phân tích & dự đoán kết quả học tập sinh viên</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='subtitle'>Dữ liệu UCI Student Performance – kết hợp khai phá cụm, luật và mô hình phân lớp để cảnh báo sớm nguy cơ trượt môn.</div>",
        unsafe_allow_html=True,
    )

