import os
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. ĐƯỜNG DẪN DỮ LIỆU
path = r"F:\tài liệu\New folder\Liver_disease_data.csv"

try:
    print("--- ĐANG KHỞI CHẠY QUY TRÌNH HUẤN LUYỆN TỐI ƯU ---")
    
    # Đọc dữ liệu với encoding để tránh lỗi font
    df = pd.read_csv(path, encoding='latin1')
    print("✅ Bước 1: Đọc dữ liệu thành công.")
    print("Danh sách cột hiện có:", df.columns.tolist())

    # 2. TIỀN XỬ LÝ DỮ LIỆU (ĐÃ SỬA LỖI CỘT)
    # Danh sách các cột cần loại bỏ nếu tồn tại
    # Chúng ta dùng list comprehension để lọc ra những cột thực sự có trong file
    cols_to_drop = ['PatientID', 'Diagnosis', 'ID', 'patientid'] 
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    
    # Tách X (đặc trưng) và y (nhãn)
    X = df.drop(columns=existing_drops).values 
    y = df['Diagnosis'].values
    
    print(f"✅ Bước 2: Đã loại bỏ cột: {existing_drops}")
    print(f"Sử dụng {X.shape[1]} cột để huấn luyện mô hình.")

    # 3. CHIA DỮ LIỆU (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. HUẤN LUYỆN MÔ HÌNH LIGHTGBM
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    print("✅ Bước 3: Mô hình LightGBM đã huấn luyện xong.")

    # 5. ĐÁNH GIÁ ĐỘ CHÍNH XÁC
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Độ chính xác mô hình: {acc*100:.2f}%")

    # 6. XUẤT FILE MÔ HÌNH (.PKL)
    # File này sẽ được lưu ngay cùng thư mục với file huan_luyen.py
    joblib.dump(model, "liver_model.pkl")
    
    # Vẽ biểu đồ độ quan trọng (Feature Importance)
    plt.figure(figsize=(10, 8))
    lgb.plot_importance(model, importance_type='gain')
    plt.title("Do quan trong cua cac chi so")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    print("-" * 50)
    print("🎉 CHÚC MỪNG! BẠN ĐÃ TẠO FILE THÀNH CÔNG.")
    print(f"📍 File mô hình: {os.path.abspath('liver_model.pkl')}")
    print("--- BÂY GIỜ BẠN CÓ THỂ CHẠY GIAO DIỆN STREAMLIT ---")

except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file CSV tại đường dẫn: {path}")
except Exception as e:
    print(f"❌ Lỗi hệ thống: {e}")