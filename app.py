import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf  # Thư viện dữ liệu tài chính số 1 thế giới

# --- Cấu hình giao diện Web ---
st.set_page_config(page_title="Dự đoán Chứng khoán VN", layout="wide")
st.title("📈 Ứng dụng Dự đoán Chứng khoán Việt Nam (Mô hình LSTM)")
st.write("Sử dụng thư viện Yahoo Finance")

# --- Sidebar (Thanh công cụ bên trái) ---
st.sidebar.header("Cấu hình hệ thống")
ticker = st.sidebar.text_input("Mã cổ phiếu (VD: VCB, FPT, HPG):", "VCB").upper()
lookback_days = 50
epochs = st.sidebar.slider("Epochs (Số vòng huấn luyện):", 5, 50, 10)
years_to_fetch = st.sidebar.slider("Số năm dữ liệu lịch sử:", 1, 5, 2)

@st.cache_data
def get_stock_data_yfinance(ticker, years=2):
    """ Hàm lấy dữ liệu siêu tốc bằng Yahoo Finance """
    try:
        # Yahoo Finance quy định mã CK Việt Nam phải có đuôi .VN
        yf_ticker = f"{ticker}.VN"
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=years*365)
        
        # Tải dữ liệu từ Yahoo Finance
        df = yf.download(yf_ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if df is None or df.empty:
            return None, f"Không tìm thấy mã {ticker} trên Yahoo Finance."
            
        # Reset index để cột Date trở thành cột dữ liệu bình thường
        df = df.reset_index()
        
        # Xử lý trường hợp yfinance trả về MultiIndex (Bản cập nhật mới của yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Đảm bảo tên cột viết hoa chữ cái đầu (Date, Open, High, Low, Close, Volume)
        cols_map = {c: c.capitalize() for c in df.columns if c.lower() in ['date', 'open', 'high', 'low', 'close', 'volume']}
        df = df.rename(columns=cols_map)
        
        # Lọc đúng 6 cột cần thiết
        needed = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in needed if c in df.columns]]
        
        # Xóa dữ liệu lỗi (NaN)
        df = df.dropna()
        return df, "Thành công"
        
    except Exception as e:
        return None, str(e)

# --- SỰ KIỆN KHI BẤM NÚT HUẤN LUYỆN ---
if st.sidebar.button("Bắt đầu lấy dữ liệu & Huấn luyện"):
    with st.spinner(f"Đang tải dữ liệu mã {ticker} từ Yahoo Finance..."):
        df, error_msg = get_stock_data_yfinance(ticker, years_to_fetch)
    
    if df is not None and len(df) > lookback_days:
        st.success(f"Tải thành công {len(df)} ngày giao dịch hợp lệ!")
        st.subheader("Dữ liệu lịch sử (5 ngày gần nhất)")
        st.dataframe(df.tail(5))
        
        # --- QUY TRÌNH HỌC MÁY (LSTM) BÁM SÁT ĐỒ ÁN ---
        data = df[['Close']].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        train_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_len]
        
        x_train, y_train = [], []
        for i in range(lookback_days, len(train_data)):
            x_train.append(train_data[i-lookback_days:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Xây dựng mô hình
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Huấn luyện
        with st.spinner(f"Đang huấn luyện AI (Epochs: {epochs}). Vui lòng chờ..."):
            model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)
        st.success("Huấn luyện mô hình hoàn tất!")
            
        # Chuẩn bị Test
        test_data = scaled_data[train_len - lookback_days:]
        x_test = []
        for i in range(lookback_days, len(test_data)):
            x_test.append(test_data[i-lookback_days:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        # Dự đoán
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        
        # --- TRỰC QUAN HÓA BẰNG PLOTLY ---
        st.subheader("Biểu đồ So sánh Giá Thực tế và Giá Dự đoán")
        valid = df[train_len:].copy()
        valid['Predictions'] = predictions
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'][:train_len], y=df['Close'][:train_len], mode='lines', name='Dữ liệu Huấn luyện (Train)'))
        fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Close'], mode='lines', name='Giá Thực tế (Test)', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=valid['Date'], y=valid['Predictions'], mode='lines', name='Giá Dự đoán (Predict)', line=dict(color='red')))
        fig.update_layout(title=f'Biểu đồ biến động giá cổ phiếu {ticker}', xaxis_title='Thời gian', yaxis_title='Giá Đóng cửa (VND)')
        st.plotly_chart(fig, use_container_width=True)

        # --- DỰ ĐOÁN TƯƠNG LAI ---
        last_50_days = scaled_data[-lookback_days:]
        X_future = np.array([last_50_days])
        X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
        pred_future = model.predict(X_future)
        pred_future_price = scaler.inverse_transform(pred_future)

        st.info(f"🔮 **Dự đoán giá đóng cửa phiên giao dịch tiếp theo cho {ticker}: {pred_future_price[0][0]:,.0f} VND**")
    else:
        st.error(f"Lỗi truy xuất: {error_msg}")