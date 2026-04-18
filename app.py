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
import yfinance as yf  # Thư viện dữ liệu tài chính

# --- Cấu hình giao diện Web ---
st.set_page_config(page_title="Dự đoán Chứng khoán VN", layout="wide")
st.title("Ứng dụng Dự đoán Chứng khoán Việt Nam (Mô hình LSTM (Long Short-Term Memory))")
st.write("Sử dụng thư viện Yahoo Finance")

# --- Sidebar (Thanh công cụ bên trái) ---
st.sidebar.header("Cấu hình hệ thống")
ticker = st.sidebar.text_input("Mã cổ phiếu (VD: VCB, FPT, HPG):", "VCB").upper()
lookback_days = 50
epochs = st.sidebar.slider("Epochs (Số vòng huấn luyện):", 5, 50, 10)
years_to_fetch = st.sidebar.slider("Số năm dữ liệu lịch sử:", 1, 20, 2)
n_future_days = st.sidebar.slider("Số phiên dự đoán tiếp theo:", 1, 30, 10)

@st.cache_data
def get_stock_data_yfinance(ticker, years=2):
    """ Hàm lấy dữ liệu bằng Yahoo Finance """
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

@st.cache_data
def get_fundamental_data(ticker):
    """ Lấy các chỉ số tài chính cơ bản của doanh nghiệp """
    try:
        yf_ticker_obj = yf.Ticker(f"{ticker}.VN")
        info = yf_ticker_obj.info

        # Trích xuất các chỉ số quan trọng, dùng .get để tránh lỗi nếu thiếu dữ liệu
        fundamentals = {
            "Tên Công ty": info.get('longName', 'N/A'),
            "Ngành": info.get('industry', 'N/A'),
            "Website": info.get('website', 'N/A'),
            "Vốn hóa thị trường": info.get('marketCap', 0),
            "P/E": info.get('trailingPE', 0),
            "P/B": info.get('priceToBook', 0),
            "ROE": info.get('returnOnEquity', 0),
            "Nợ/Vốn chủ sở hữu (D/E)": info.get('debtToEquity', 0),
            "EPS": info.get('trailingEps', 0),
            "Cổ tức/Thị giá (TB 5 năm)": info.get('fiveYearAvgDividendYield', 0),
        }
        return fundamentals
    except Exception:
        return None

# --- SỰ KIỆN KHI BẤM NÚT HUẤN LUYỆN ---
if st.sidebar.button("Bắt đầu lấy dữ liệu & Huấn luyện"):
    with st.spinner(f"Đang tải dữ liệu mã {ticker} từ Yahoo Finance..."):
        df, error_msg = get_stock_data_yfinance(ticker, years_to_fetch)
    
    if df is not None and len(df) > lookback_days:
        st.success(f"Tải thành công {len(df)} ngày giao dịch hợp lệ!")

        # --- HIỂN THỊ PHÂN TÍCH CƠ BẢN ---
        st.subheader(f"Phân tích cơ bản doanh nghiệp: {ticker}")
        fundamentals = get_fundamental_data(ticker)
        if fundamentals:
            st.write(f"**{fundamentals['Tên Công ty']}** ({fundamentals['Ngành']}) - [Website]({fundamentals['Website']})")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Vốn hóa thị trường (tỷ VND)", f"{fundamentals['Vốn hóa thị trường']/1e9:,.0f}")
            col2.metric("P/E (Giá/Lợi nhuận)", f"{fundamentals['P/E']:.2f}" if fundamentals['P/E'] else "N/A", help="Giá bạn trả cho mỗi đồng lợi nhuận. Càng thấp càng tốt.")
            col3.metric("P/B (Giá/Giá trị sổ sách)", f"{fundamentals['P/B']:.2f}" if fundamentals['P/B'] else "N/A", help="So sánh giá thị trường với giá trị sổ sách của công ty.")
            col4.metric("ROE (Lợi nhuận/Vốn CSH)", f"{fundamentals['ROE']*100:.2f}%" if fundamentals['ROE'] else "N/A", help="Hiệu quả sử dụng vốn của cổ đông. Càng cao càng tốt.")
            col5.metric("EPS (Lợi nhuận/Cổ phiếu)", f"{fundamentals['EPS']:,.0f}" if fundamentals['EPS'] else "N/A", help="Lợi nhuận trên mỗi cổ phiếu. Càng cao càng tốt.")
            col6.metric("Cổ tức/Thị giá (TB 5 năm)", f"{fundamentals['Cổ tức/Thị giá (TB 5 năm)']:.2f}%" if fundamentals['Cổ tức/Thị giá (TB 5 năm)'] else "N/A", help="Tỷ suất cổ tức trung bình trong 5 năm gần nhất. Càng cao càng hấp dẫn cho đầu tư dài hạn.")

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
        # Tạo biểu đồ hiển thị dữ liệu lịch sử và dự đoán tương lai
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Dữ liệu Lịch sử'))
        
        st.subheader(f"Bảng dự đoán {n_future_days} phiên giao dịch tiếp theo")

        # Lấy lookback_days ngày cuối cùng từ dữ liệu đã scale
        last_sequence = scaled_data[-lookback_days:]
        future_predictions_scaled = []
        current_batch = last_sequence.reshape(1, lookback_days, 1)

        # Vòng lặp dự đoán
        with st.spinner(f"Đang dự đoán {n_future_days} phiên tiếp theo..."):
            for i in range(n_future_days):
                # Dự đoán 1 ngày tiếp theo
                next_prediction_scaled = model.predict(current_batch, verbose=0)[0]
                
                # Lưu lại kết quả dự đoán
                future_predictions_scaled.append(next_prediction_scaled)
                
                # Cập nhật lại batch đầu vào: bỏ ngày cũ nhất, thêm ngày mới dự đoán
                current_batch = np.append(current_batch[:, 1:, :], [[next_prediction_scaled]], axis=1)

        # Chuyển đổi các giá trị dự đoán về thang đo ban đầu
        future_predictions = scaler.inverse_transform(future_predictions_scaled)

        # Tạo các ngày trong tương lai (chỉ lấy ngày làm việc)
        last_date = df['Date'].iloc[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_future_days)

        # Tạo DataFrame cho kết quả dự đoán và hiển thị
        df_future = pd.DataFrame(data={'Ngày': future_dates, 'Giá đóng cửa dự đoán (VND)': future_predictions.flatten()})
        df_future['Ngày'] = df_future['Ngày'].dt.strftime('%Y-%m-%d')
        df_future['Giá đóng cửa dự đoán (VND)'] = df_future['Giá đóng cửa dự đoán (VND)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_future, use_container_width=True)

        # Thêm dự đoán tương lai vào biểu đồ
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions.flatten(), mode='lines', name=f'Dự đoán {n_future_days} ngày tới', line=dict(color='green', dash='dash')))
        
        # Cập nhật layout và hiển thị biểu đồ
        fig.update_layout(title=f'Biểu đồ biến động và dự đoán giá cổ phiếu {ticker}', xaxis_title='Thời gian', yaxis_title='Giá Đóng cửa (VND)')
        st.plotly_chart(fig, use_container_width=True)

        # Hiển thị dự đoán cho ngày đầu tiên trong chuỗi
        if n_future_days > 0:
            first_future_price = future_predictions.flatten()[0]
            st.info(f" **Dự đoán giá đóng cửa phiên giao dịch tiếp theo cho {ticker}: {first_future_price:,.0f} VND**")
    else:
        st.error(f"Lỗi truy xuất: {error_msg}")