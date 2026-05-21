import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Quản lý Gan AI", 
    layout="wide",
    page_icon="🫀",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH GIAO DIỆN ---
st.markdown("""
<style>
    /* Import font đẹp */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    
    /* Card đẹp */
    .custom-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    
    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid;
    }
    
    /* Button đẹp */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -5px rgba(102,126,234,0.4);
    }
    
    /* Sidebar đẹp */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.8) !important;
    }
    
    /* Tabs đẹp */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f0f2f6 0%, #e4e7ec 100%);
        border-radius: 12px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Expander đẹp */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        font-weight: 600;
    }
    
    /* Divider gradient */
    .gradient-divider {
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        height: 3px;
        border-radius: 3px;
        margin: 1.5rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        color: white;
        margin-top: 2rem;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    .badge-success { background: #d4edda; color: #155724; }
    .badge-warning { background: #fff3cd; color: #856404; }
    .badge-danger { background: #f8d7da; color: #721c24; }
    .badge-info { background: #d1ecf1; color: #0c5460; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ĐẸP ---
st.markdown("""
<div class="main-header">
    <h1>🫀 Hệ thống Hỗ trợ Chẩn đoán Gan nhiễm mỡ</h1>
    <p>Powered by LightGBM & Random Forest | Độ chính xác lên đến 89%</p>
</div>
""", unsafe_allow_html=True)

# --- MENU ĐIỀU HƯỚNG (Sidebar đã có CSS gradient) ---
st.sidebar.title("✨ Danh mục hệ thống")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Chọn chức năng:", 
    ["🩺 Dự đoán & Xem kết quả", "📊 Phân tích chuyên sâu", "📅 Đăng ký & Thanh toán", "📚 Cẩm nang & Thư viện", "⚖️ So sánh mô hình"],
    format_func=lambda x: f"**{x}**"
)

st.sidebar.markdown("---")
st.sidebar.info("📌 **Thông tin hệ thống**\n\n- Mô hình: LightGBM + RF\n- Dữ liệu: 1.700 bệnh nhân\n- Accuracy: 88.53%")
st.sidebar.caption("© 2024 - Hệ thống hỗ trợ y tế thông minh")

# --- LOAD MÔ HÌNH ---
@st.cache_resource
def load_my_model():
    if os.path.exists("liver_model.pkl"):
        return joblib.load("liver_model.pkl")
    return None

@st.cache_resource
def load_rf_model():
    if os.path.exists("rf_model.pkl"):
        return joblib.load("rf_model.pkl")
    return None

model = load_my_model()
model_rf = load_rf_model()

# ==================== CHỨC NĂNG 1: DỰ ĐOÁN ====================
if menu == "🩺 Dự đoán & Xem kết quả":
    st.markdown("### 🩺 Chẩn đoán Gan nhiễm mỡ bằng LightGBM")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Không tìm thấy file 'liver_model.pkl'. Hãy chạy file huấn luyện trước!")
    else:
        col_info, col_form = st.columns([1, 2])
        
        with col_info:
            st.markdown("""
            <div class="custom-card">
                <h4>📋 Hướng dẫn</h4>
                <p>Nhập đầy đủ 10 chỉ số lâm sàng để hệ thống đưa ra dự đoán chính xác nhất.</p>
                <hr>
                <p><span class="badge badge-info">💡 Gợi ý</span></p>
                <ul>
                    <li>BMI = Cân nặng(kg) / (Chiều cao(m))²</li>
                    <li>Chỉ số gan bình thường: 20-40</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_form:
            with st.expander("📝 Nhập thông số lâm sàng", expanded=True):
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input("🎂 Độ tuổi", min_value=6, max_value=100, value=45, step=1)
                    gender_val = st.selectbox("⚧ Giới tính", [0, 1], format_func=lambda x: "👨 Nam" if x==0 else "👩 Nữ")
                    bmi = st.number_input("📏 Chỉ số BMI", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
                    alcohol = st.slider("🍺 Lượng rượu/tuần", 0.0, 20.0, 5.0, step=0.5)
                    smoking_val = st.selectbox("🚬 Hút thuốc", [0, 1], format_func=lambda x: "❌ Không" if x==0 else "✅ Có")
                with c2:
                    genetic_val = st.selectbox("🧬 Di truyền", [0, 1, 2], format_func=lambda x: ["🟢 Thấp", "🟡 Trung bình", "🔴 Cao"][x])
                    physical = st.slider("🏃 Vận động (giờ/tuần)", 0.0, 10.0, 3.0, step=0.5)
                    diabetes_val = st.selectbox("🩸 Tiểu đường", [0, 1], format_func=lambda x: "❌ Không" if x==0 else "✅ Có")
                    hypertension_val = st.selectbox("💓 Cao huyết áp", [0, 1], format_func=lambda x: "❌ Không" if x==0 else "✅ Có")
                    liver_test = st.number_input("🔬 Chỉ số Gan", min_value=10.0, max_value=100.0, value=50.0, step=1.0)
                
                submit = st.button("🚀 Dự đoán kết quả", use_container_width=True)

        if submit:
            input_data = np.array([[age, gender_val, bmi, alcohol, smoking_val, genetic_val, physical, diabetes_val, hypertension_val, liver_test]])
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            # Hiển thị bảng dữ liệu
            with st.expander("📊 Xem lại dữ liệu đã nhập", expanded=True):
                data_display = {
                    "Chỉ số": ["Độ tuổi", "Giới tính", "BMI", "Rượu/tuần", "Hút thuốc", "Di truyền", "Vận động", "Tiểu đường", "Huyết áp", "Chỉ số Gan"],
                    "Giá trị": [age, "Nam" if gender_val==0 else "Nữ", bmi, f"{alcohol} đơn vị", "Có" if smoking_val==1 else "Không", ["Thấp", "Trung bình", "Cao"][genetic_val], f"{physical} giờ", "Có" if diabetes_val==1 else "Không", "Có" if hypertension_val==1 else "Không", liver_test]
                }
                st.dataframe(pd.DataFrame(data_display), use_container_width=True, hide_index=True)
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # Kết quả dự đoán
            col_res1, col_res2 = st.columns([1, 1])
            
            with col_res1:
                if prediction == 1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 20px; padding: 1.5rem; text-align: center;">
                        <h2 style="color: white;">⚠️ CÓ NGUY CƠ MẮC BỆNH</h2>
                        <p style="color: white; font-size: 2rem; font-weight: bold;">{:.2f}%</p>
                        <p style="color: white;">Tỷ lệ mắc bệnh dự đoán</p>
                    </div>
                    """.format(prob*100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 20px; padding: 1.5rem; text-align: center;">
                        <h2 style="color: white;">✅ AN TOÀN</h2>
                        <p style="color: white; font-size: 2rem; font-weight: bold;">{:.2f}%</p>
                        <p style="color: white;">Độ tin cậy</p>
                    </div>
                    """.format((1-prob)*100), unsafe_allow_html=True)
            
            with col_res2:
                # Thanh tiến trình đẹp
                st.markdown("#### 📊 Mức độ nguy cơ")
                st.progress(prob)
                st.caption(f"Ngưỡng nguy cơ: {prob*100:.1f}%")
                
                # Gauge meter bằng plotly
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob*100,
                    title = {'text': "Mức độ nguy cơ (%)"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#ff6b6b"},
                        'steps': [
                            {'range': [0, 30], 'color': '#d4edda'},
                            {'range': [30, 70], 'color': '#fff3cd'},
                            {'range': [70, 100], 'color': '#f8d7da'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            # Liệu pháp
            st.markdown("### 💊 Liệu pháp đề xuất")
            col_t1, col_t2 = st.columns(2)
            
            if prediction == 1:
                with col_t1:
                    st.info("**📋 Phác đồ điều trị**\n\n- Siêu âm đàn hồi gan định kỳ\n- Xét nghiệm men gan chuyên sâu\n- Dùng thuốc hỗ trợ tế bào gan\n- Theo dõi chỉ số mỡ máu")
                with col_t2:
                    st.warning("**🥗 Chế độ sinh hoạt**\n\n- Giảm cân cấp thiết (5-10% trọng lượng)\n- Ngừng rượu bia hoàn toàn\n- Tập thể dục 30 phút/ngày\n- Chế độ ăn ít dầu mỡ, nhiều rau xanh")
            else:
                with col_t1:
                    st.success("**📋 Theo dõi định kỳ**\n\n- Khám sức khỏe 6 tháng/lần\n- Tầm soát xơ hóa gan\n- Tiêm chủng đầy đủ\n- Kiểm tra men gan định kỳ")
                with col_t2:
                    st.info("**🥗 Lối sống lành mạnh**\n\n- Hạn chế đồ chiên xào, dầu mỡ\n- Uống đủ 2 lít nước/ngày\n- Giảm căng thẳng, ngủ đủ giấc\n- Bổ sung chất xơ và vitamin")

# ==================== CHỨC NĂNG 2: PHÂN TÍCH ====================
elif menu == "📊 Phân tích chuyên sâu":
    st.markdown("### 📊 Phân tích ảnh hưởng các chỉ số")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    col_des, col_chart = st.columns([1, 2])
    
    with col_des:
        st.markdown("""
        <div class="custom-card">
            <h4>🔍 Giải thích biểu đồ</h4>
            <p>Biểu đồ thể hiện mức độ ảnh hưởng của từng chỉ số đến bệnh gan nhiễm mỡ dựa trên phân tích từ mô hình LightGBM.</p>
            <hr>
            <p><span class="badge badge-info">📌 Ghi chú</span></p>
            <ul>
                <li><strong>LiverFunctionTest</strong> là yếu tố quan trọng nhất (24.3%)</li>
                <li><strong>BMI</strong> đứng thứ hai (12.1%)</li>
                <li><strong>PhysicalActivity</strong> có ảnh hưởng tích cực (giảm nguy cơ)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_chart:
        chart_data = pd.DataFrame({
            'Chỉ số': ['LiverFunction', 'BMI', 'Age', 'Physical Activity', 'Alcohol', 'Genetic', 'Diabetes', 'Hypertension', 'Smoking', 'Gender'],
            'Mức độ ảnh hưởng (%)': [24.3, 12.1, 10.7, 10.5, 9.2, 8.4, 7.6, 6.8, 6.4, 3.6]
        })
        
        fig = px.bar(chart_data, x='Mức độ ảnh hưởng (%)', y='Chỉ số', orientation='h',
                     color='Mức độ ảnh hưởng (%)', color_continuous_scale='Viridis',
                     title='🏆 Mức độ ảnh hưởng của các chỉ số đến bệnh gan nhiễm mỡ',
                     text='Mức độ ảnh hưởng (%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📈 Thống kê phân phối dữ liệu")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric("📊 Tổng số mẫu", "1.700", delta=None)
    with col_s2:
        st.metric("🩺 Bệnh nhân có bệnh", "765", delta="45%")
    with col_s3:
        st.metric("✅ Người khỏe mạnh", "935", delta="55%")
    with col_s4:
        st.metric("🎯 Độ chính xác", "88.53%", delta="+2.1%")

# ==================== CHỨC NĂNG 3: ĐĂNG KÝ ====================
elif menu == "📅 Đăng ký & Thanh toán":
    st.markdown("### 📅 Quản lý đăng ký lịch khám")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    tab_reg, tab_cancel = st.tabs(["🆕 Đăng ký mới", "❌ Hủy lịch khám"])
    
    with tab_reg:
        with st.form("booking_form", border=True):
            st.markdown("#### 👤 Thông tin bệnh nhân")
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Họ và tên", placeholder="Nhập họ tên đầy đủ")
            with c2:
                phone = st.text_input("Số điện thoại", placeholder="0987xxxxxx")
            
            date_visit = st.date_input("Chọn ngày khám", min_value=datetime.now())
            
            st.markdown("#### 💳 Thanh toán phí dịch vụ")
            service = st.selectbox("Gói dịch vụ", ["🏥 Khám cơ bản (200.000 VNĐ)", "🔬 Tầm soát Gan chuyên sâu (500.000 VNĐ)", "💎 Gói VIP (1.200.000 VNĐ)"])
            pay_method = st.radio("Phương thức thanh toán", ["🏦 Chuyển khoản ngân hàng", "📱 Ví điện tử (Momo/ZaloPay)", "💳 Thẻ tín dụng"], horizontal=True)
            
            btn_pay = st.form_submit_button("💳 Xác nhận đăng ký & Thanh toán", use_container_width=True)
        
        if btn_pay:
            if name and phone:
                booking_code = f"GAN-{random.randint(1000, 9999)}"
                st.balloons()
                st.success(f"✅ Đăng ký thành công cho bệnh nhân: **{name}**")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-radius: 15px; padding: 1rem; margin-top: 1rem;">
                    <h4>🧾 HÓA ĐƠN DỊCH VỤ</h4>
                    <p>🔖 <strong>Mã đặt lịch:</strong> {booking_code}</p>
                    <p>👤 <strong>Bệnh nhân:</strong> {name}</p>
                    <p>📅 <strong>Ngày hẹn:</strong> {date_visit.strftime('%d/%m/%Y')}</p>
                    <p>💼 <strong>Dịch vụ:</strong> {service}</p>
                    <p>💳 <strong>Trạng thái:</strong> Đã thanh toán qua {pay_method}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("⚠️ Vui lòng nhập đầy đủ Họ tên và Số điện thoại!")
    
    with tab_cancel:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("❌ Yêu cầu hủy lịch và Hoàn phí")
        cancel_id = st.text_input("Nhập Mã đặt lịch", placeholder="Ví dụ: GAN-1234")
        cancel_phone = st.text_input("Số điện thoại đăng ký", placeholder="0987xxxxxx")
        cancel_reason = st.text_area("Lý do hủy lịch", placeholder="Vui lòng nêu rõ lý do...")
        
        if st.button("🗑️ Xác nhận hủy lịch", use_container_width=True):
            if cancel_id and cancel_phone:
                st.warning(f"🔔 Đã nhận yêu cầu hủy lịch cho mã: **{cancel_id}**")
                st.info("Hệ thống sẽ xác thực và hoàn phí (nếu có) trong vòng 24h làm việc.")
            else:
                st.error("⚠️ Vui lòng nhập Mã đặt lịch và Số điện thoại để tiếp tục!")
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== CHỨC NĂNG 4: CẨM NANG ====================
elif menu == "📚 Cẩm nang & Thư viện":
    st.markdown("### 📚 Thư viện Y khoa & Cẩm nang Sức khỏe")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="custom-card">
            <h4>🍏 Chế độ Dinh dưỡng</h4>
            <ul>
                <li><strong>❌ Hạn chế:</strong> Dầu mỡ, mỡ động vật, đường tinh luyện, đồ chiên xào</li>
                <li><strong>✅ Tăng cường:</strong> Rau xanh, trái cây, ngũ cốc nguyên hạt</li>
                <li><strong>🥩 Thực phẩm nên dùng:</strong> Cá hồi, ức gà, đậu phụ, các loại hạt</li>
                <li><strong>💧 Đồ uống:</strong> Trà xanh, nước lọc (2 lít/ngày), cà phê vừa phải</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card">
            <h4>🏃 Chế độ Vận động</h4>
            <ul>
                <li><strong>⏰ Thời gian:</strong> Ít nhất 30 phút mỗi ngày</li>
                <li><strong>🏊 Hình thức:</strong> Đi bộ nhanh, bơi lội, đạp xe, yoga</li>
                <li><strong>📅 Tần suất:</strong> 5 ngày/tuần</li>
                <li><strong>💪 Cường độ:</strong> Vừa sức, tim đập nhanh nhẹ</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📖 Thư viện Sách Y khoa")
    
    if 'medical_books' not in st.session_state:
        st.session_state.medical_books = pd.DataFrame([
            {"Tên sách": "Cẩm nang Gan mật 2024", "Tác giả": "Bộ Y tế", "Liên kết": "https://moh.gov.vn", "Ngày cập nhật": "2024-01-01"},
            {"Tên sách": "LightGBM trong Y học", "Tác giả": "AI Health", "Liên kết": "https://github.com", "Ngày cập nhật": "2024-03-15"},
            {"Tên sách": "Dinh dưỡng cho người bệnh gan", "Tác giả": "PGS.TS Nguyễn Thị Lâm", "Liên kết": "#", "Ngày cập nhật": "2024-02-10"}
        ])
    
    edited_df = st.data_editor(st.session_state.medical_books, num_rows="dynamic", use_container_width=True, 
                                column_config={"Liên kết": st.column_config.LinkColumn("Đường dẫn Online")})
    
    if st.button("💾 Lưu thay đổi", use_container_width=True):
        st.session_state.medical_books = edited_df
        st.success("✅ Đã cập nhật danh sách thành công!")
    
    st.markdown("### 🎥 Video hướng dẫn")
    st.video("https://www.youtube.com/watch?v=kYI_U7uQ038")

# ==================== CHỨC NĂNG 5: SO SÁNH MÔ HÌNH ====================
else:
    st.markdown("### ⚖️ So sánh song song: LightGBM vs Random Forest")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if model:
            st.success("✅ **LightGBM**: Đã sẵn sàng")
        else:
            st.error("❌ LightGBM chưa được tải")
    with col_status2:
        if model_rf:
            st.success("✅ **Random Forest**: Đã sẵn sàng")
        else:
            st.error("❌ Random Forest chưa được tải")
    
    with st.form("compare_form", border=True):
        st.subheader("📝 Nhập dữ liệu lâm sàng dùng chung")
        c1, c2 = st.columns(2)
        with c1:
            c_age = st.number_input("🎂 Độ tuổi", 6, 100, 45, key="comp_age")
            c_gen = st.selectbox("⚧ Giới tính", [0, 1], format_func=lambda x: "Nam" if x==0 else "Nữ", key="comp_gender")
            c_bmi = st.number_input("📏 BMI", 10.0, 50.0, 24.0, step=0.1, key="comp_bmi")
            c_alc = st.slider("🍺 Rượu/tuần", 0.0, 20.0, 5.0, step=0.5, key="comp_alcohol")
            c_smk = st.selectbox("🚬 Hút thuốc", [0, 1], format_func=lambda x: "Không" if x==0 else "Có", key="comp_smoking")
        with c2:
            c_gtr = st.selectbox("🧬 Di truyền", [0, 1, 2], format_func=lambda x: ["Thấp", "Trung bình", "Cao"][x], key="comp_genetic")
            c_phy = st.slider("🏃 Vận động (giờ/tuần)", 0.0, 10.0, 3.0, step=0.5, key="comp_physical")
            c_dia = st.selectbox("🩸 Tiểu đường", [0, 1], format_func=lambda x: "Không" if x==0 else "Có", key="comp_diabetes")
            c_hyp = st.selectbox("💓 Cao huyết áp", [0, 1], format_func=lambda x: "Không" if x==0 else "Có", key="comp_hypertension")
            c_liv = st.number_input("🔬 Chỉ số Gan", 10.0, 100.0, 50.0, step=1.0, key="comp_liver")
        
        btn_compare = st.form_submit_button("⚡ So sánh song song", use_container_width=True)
    
    if btn_compare:
        if model is None and model_rf is None:
            st.error("❌ Không có mô hình nào được tải!")
        else:
            input_compare = np.array([[c_age, c_gen, c_bmi, c_alc, c_smk, c_gtr, c_phy, c_dia, c_hyp, c_liv]])
            
            col_lgbm, col_vs, col_rf = st.columns([2, 1, 2])
            
            with col_lgbm:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 1rem; text-align: center;">
                    <h3 style="color: white;">🔵 LightGBM</h3>
                </div>
                """, unsafe_allow_html=True)
                if model:
                    p_lgbm = model.predict_proba(input_compare)[0][1]
                    pred_lgbm = model.predict(input_compare)[0]
                    st.metric("Xác suất mắc bệnh", f"{p_lgbm*100:.2f}%")
                    st.progress(p_lgbm)
                    if pred_lgbm == 1:
                        st.error("🚨 **Có nguy cơ mắc bệnh**")
                    else:
                        st.success("✅ **An toàn**")
                else:
                    st.warning("⚠️ Chưa tải mô hình")
            
            with col_vs:
                st.markdown("<h2 style='text-align: center; padding-top: 80px;'>🆚 VS</h2>", unsafe_allow_html=True)
            
            with col_rf:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 15px; padding: 1rem; text-align: center;">
                    <h3 style="color: white;">🟢 Random Forest</h3>
                </div>
                """, unsafe_allow_html=True)
                if model_rf:
                    p_rf = model_rf.predict_proba(input_compare)[0][1]
                    pred_rf = model_rf.predict(input_compare)[0]
                    st.metric("Xác suất mắc bệnh", f"{p_rf*100:.2f}%")
                    st.progress(p_rf)
                    if pred_rf == 1:
                        st.error("🚨 **Có nguy cơ mắc bệnh**")
                    else:
                        st.success("✅ **An toàn**")
                else:
                    st.warning("⚠️ Chưa tải mô hình")
            
            if model and model_rf:
                st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
                st.subheader("📊 Biểu đồ so sánh")
                
                comp_data = pd.DataFrame({
                    'Mô hình': ['LightGBM', 'Random Forest'],
                    'Xác suất (%)': [p_lgbm*100, p_rf*100]
                })
                fig = px.bar(comp_data, x='Mô hình', y='Xác suất (%)', color='Mô hình',
                            text='Xác suất (%)', color_discrete_sequence=['#667eea', '#4facfe'])
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ĐẸP ---
st.markdown("""
<div class="footer">
    <p>🫀 Hệ thống Hỗ trợ Chẩn đoán Gan nhiễm mỡ | Powered by LightGBM & Streamlit</p>
    <p>© 2024 - Dữ liệu từ 1.700 bệnh nhân | Độ chính xác: 88.53%</p>
</div>
""", unsafe_allow_html=True)

