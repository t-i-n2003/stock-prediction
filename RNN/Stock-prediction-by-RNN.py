import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from vnstock import *
from datetime import datetime

# Lấy ngày hôm nay
today = datetime.today().strftime("%Y-%m-%d")

# Lấy dữ liệu lịch sử giá cổ phiếu
fpt = stock_historical_data('FPT', start_date='1990-01-01', end_date=today)
print("Dữ liệu tải về thành công. Số lượng dòng:", len(fpt))
print(fpt.tail(5))  # Kiểm tra cấu trúc dữ liệu

# Kiểm tra cột 'close'
if 'close' not in fpt.columns:
    raise ValueError("Cột 'close' không tồn tại trong dữ liệu!")

# Lấy giá đóng cửa
close_prices = fpt['close'].dropna().values
print("Dữ liệu cột 'close' đã xử lý: Số lượng giá trị hợp lệ:", len(close_prices))

# Chuẩn hóa dữ liệu
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices.reshape(-1, 1))
print("Dữ liệu sau khi chuẩn hóa. Kích thước:", scaled_prices.shape)

# Tạo dữ liệu chuỗi thời gian
time_steps = 5
X, y = [], []
for i in range(time_steps, len(scaled_prices)):
    X.append(scaled_prices[i-time_steps:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)
print("Dữ liệu X và y được tạo. Kích thước:")
print("X:", X.shape, "y:", y.shape)

# Định hình lại X để phù hợp với RNN
features = 1
X = X.reshape(X.shape[0], X.shape[1], features)
print("X sau khi định hình lại:", X.shape)

# Chia tập dữ liệu
split_ratio = 0.8
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print("Tập huấn luyện và kiểm tra được chia:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Xây dựng mô hình
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(time_steps, features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print("Mô hình RNN đã được xây dựng thành công.")

# Huấn luyện mô hình
try:
    model.fit(X_train, y_train, epochs=50, batch_size=16)
    print("Huấn luyện mô hình hoàn tất.")
except Exception as e:
    print("Lỗi khi huấn luyện mô hình:", e)

# Lưu mô hình và trọng số
model.save_weights('SPRNN_weight.h5')  # Lưu trọng số mô hình
model.save('SPRNN_weight.h5')  # Lưu mô hình hoàn chỉnh (cấu trúc + trọng số)
print("Mô hình và trọng số đã được lưu.")

# Dự báo
try:
    prediction = model.predict(X_test)
    print("Dự báo hoàn tất. Kết quả:")
    print(prediction[:5])  # Hiển thị 5 giá trị dự báo đầu tiên

    # Chuyển đổi kết quả dự báo về lại không gian giá trị gốc
    predicted_prices = scaler.inverse_transform(prediction)
    print("Dự báo sau khi giải mã về giá trị gốc:")
    print(predicted_prices[:5])  # Hiển thị 5 giá trị dự báo đầu tiên sau khi giải mã

except Exception as e:
    print("Lỗi khi dự báo:", e)
