import streamlit as st

from app.model_loader import encode_single_input


def get_display_mappings():
    """Các mapping giữa lựa chọn tiếng Việt và mã gốc trong dataset."""
    school_options = {
        "Trường GP": "GP",
        "Trường MS": "MS",
    }
    sex_options = {
        "Nữ": "F",
        "Nam": "M",
    }
    address_options = {
        "Thành thị": "U",
        "Nông thôn": "R",
    }
    famsize_options = {
        "Gia đình nhỏ (≤ 3 người)": "LE3",
        "Gia đình lớn (> 3 người)": "GT3",
    }
    pstatus_options = {
        "Bố mẹ sống cùng nhau": "T",
        "Bố mẹ ly thân/ly hôn": "A",
    }
    yes_no_options = {
        "Có": "yes",
        "Không": "no",
    }
    return (
        school_options,
        sex_options,
        address_options,
        famsize_options,
        pstatus_options,
        yes_no_options,
    )


def render_predict_tab(data: dict):
    """Giao diện tab dự đoán nguy cơ trượt/đỗ."""
    st.subheader("Thông tin sinh viên")

    (
        school_options,
        sex_options,
        address_options,
        famsize_options,
        pstatus_options,
        yes_no_options,
    ) = get_display_mappings()

    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            age = st.number_input("Tuổi", min_value=15, max_value=25, value=17)
            studytime = st.selectbox(
                "Thời gian học mỗi tuần (mức 1–4)",
                options=[1, 2, 3, 4],
                index=1,
            )
            failures = st.number_input(
                "Số lần trượt trước đây",
                min_value=0,
                max_value=5,
                value=0,
                help="failures trong dataset UCI",
            )
            absences = st.number_input(
                "Số buổi vắng học",
                min_value=0,
                max_value=93,
                value=3,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            school_label = st.selectbox(
                "Trường", options=list(school_options.keys())
            )
            sex_label = st.selectbox("Giới tính", options=list(sex_options.keys()))
            address_label = st.selectbox(
                "Địa chỉ gia đình", options=list(address_options.keys())
            )
            famsize_label = st.selectbox(
                "Quy mô gia đình", options=list(famsize_options.keys())
            )
            Pstatus_label = st.selectbox(
                "Tình trạng sống cùng bố mẹ",
                options=list(pstatus_options.keys()),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            Dalc = st.slider("Mức uống rượu ngày thường (1–5)", 1, 5, 1)
            Walc = st.slider("Mức uống rượu cuối tuần (1–5)", 1, 5, 1)
            G1 = st.slider("Điểm kỳ 1 (G1)", 0, 20, 10)
            G2 = st.slider("Điểm kỳ 2 (G2)", 0, 20, 10)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Các hỗ trợ / hoạt động và môi trường")
    with st.container():
        cols_sup = st.columns(4)
        with cols_sup[0]:
            schoolsup_label = st.selectbox(
                "Học phụ đạo (schoolsup)",
                list(yes_no_options.keys()),
                index=1,
            )
        with cols_sup[1]:
            famsup_label = st.selectbox(
                "Hỗ trợ học tập từ gia đình (famsup)",
                list(yes_no_options.keys()),
                index=1,
            )
        with cols_sup[2]:
            paid_label = st.selectbox(
                "Lớp thêm trả phí (paid)",
                list(yes_no_options.keys()),
                index=1,
            )
        with cols_sup[3]:
            activities_label = st.selectbox(
                "Hoạt động ngoại khóa (activities)",
                list(yes_no_options.keys()),
                index=1,
            )

        cols_more = st.columns(4)
        with cols_more[0]:
            nursery_label = st.selectbox(
                "Đã học mầm non (nursery)",
                list(yes_no_options.keys()),
                index=1,
            )
        with cols_more[1]:
            higher_label = st.selectbox(
                "Có ý định học tiếp (higher)",
                list(yes_no_options.keys()),
                index=0,
            )
        with cols_more[2]:
            internet_label = st.selectbox(
                "Có internet ở nhà",
                list(yes_no_options.keys()),
                index=0,
            )
        with cols_more[3]:
            romantic_label = st.selectbox(
                "Đang hẹn hò (romantic)",
                list(yes_no_options.keys()),
                index=1,
            )

    with st.container():
        st.markdown("#### Yếu tố gia đình và xã hội")
        famrel = st.slider("Mối quan hệ gia đình (1–5)", 1, 5, 4)
        freetime = st.slider("Thời gian rảnh (1–5)", 1, 5, 3)
        goout = st.slider("Mức độ đi chơi với bạn (1–5)", 1, 5, 3)
        health = st.slider("Đánh giá sức khỏe tổng quan (1–5)", 1, 5, 3)

    form_data = {
        "school": school_options[school_label],
        "sex": sex_options[sex_label],
        "age": age,
        "address": address_options[address_label],
        "famsize": famsize_options[famsize_label],
        "Pstatus": pstatus_options[Pstatus_label],
        "Medu": 2,
        "Fedu": 2,
        "Mjob": "other",
        "Fjob": "other",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 1,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": yes_no_options[schoolsup_label],
        "famsup": yes_no_options[famsup_label],
        "paid": yes_no_options[paid_label],
        "activities": yes_no_options[activities_label],
        "nursery": yes_no_options[nursery_label],
        "higher": yes_no_options[higher_label],
        "internet": yes_no_options[internet_label],
        "romantic": yes_no_options[romantic_label],
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2,
    }

    if st.button("Dự đoán kết quả học tập"):
        X_input = encode_single_input(
            form_data,
            data["feature_columns"],
            data["label_encoders"],
            data["scaler"],
        )
        proba = data["clf"].predict_proba(X_input)[0, 1]
        pred = int(proba >= 0.5)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(
                "Kết quả dự đoán",
                "ĐỖ" if pred == 1 else "TRƯỢT",
                help="Dựa trên mô hình Random Forest huấn luyện từ dữ liệu UCI.",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with col_res2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Xác suất đỗ (%)", f"{proba * 100:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)

        if proba < 0.4:
            level = "Nguy cơ trượt cao"
            bg = "#fee2e2"
            border = "#dc2626"
            txt = "#991b1b"
        elif proba < 0.7:
            level = "Nguy cơ trung bình"
            bg = "#fef3c7"
            border = "#d97706"
            txt = "#92400e"
        else:
            level = "Nguy cơ thấp"
            bg = "#dcfce7"
            border = "#16a34a"
            txt = "#166534"

        st.markdown(
            f"""
            <div class="risk-box" style="background-color:{bg}; border:1px solid {border}; color:{txt};">
                Mức độ rủi ro: {level}
            </div>
            """,
            unsafe_allow_html=True,
        )

