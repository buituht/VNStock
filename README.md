Bước 1: Cài đặt các thư viện cần thiết
Trước tiên, bạn cần đảm bảo đã cài đặt tất cả các thư viện mà chương trình yêu cầu. Bạn có thể cài đặt tất cả chúng bằng một lệnh duy nhất.

Mở Terminal (trên macOS/Linux) hoặc Command Prompt/PowerShell (trên Windows) và chạy lệnh sau:

bash
pip install streamlit pandas numpy plotly scikit-learn tensorflow yfinance
Lệnh này sẽ tải và cài đặt:

streamlit: Nền tảng để xây dựng ứng dụng web.
pandas, numpy: Dành cho xử lý dữ liệu.
plotly: Dành cho việc vẽ biểu đồ.
scikit-learn, tensorflow: Dành cho mô hình học máy (LSTM).
yfinance: Để tải dữ liệu chứng khoán.
Bước 2: Chạy ứng dụng
Sau khi quá trình cài đặt hoàn tất, bạn đã sẵn sàng để khởi chạy ứng dụng.

Trong cùng cửa sổ Terminal/Command Prompt, hãy di chuyển đến thư mục chứa tệp app.py của bạn bằng lệnh cd (change directory).

Chạy chương trình
streamlit run app.py
