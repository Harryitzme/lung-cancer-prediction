import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Dữ liệu từ 2020 đến 2024 



years = np.array([2020, 2021, 2022, 2023, 2024])
deaths = np.array([23797, 23860, 22597, 23106, 23800])  # Giả định tăng đều

# 2. Tạo DataFrame
data = pd.DataFrame({
    'Year': years,
    'Deaths': deaths
})

# 3. Huấn luyện mô hình hồi quy tuyến tính
X = data[['Year']]
y = data['Deaths']
model = LinearRegression()
model.fit(X, y)

# 4. Dự đoán cho năm 2025
year_2025 = np.array([[2025]])
predicted_2025 = model.predict(year_2025)[0]

print(f"\nPrediction of lung cancer deaths in 2025: {int(predicted_2025):,} case\n")

# 5. Vẽ biểu đồ
plt.figure(figsize=(10,6))
plt.plot(years, deaths, label='Actual data (2020-2024)', marker='o')
plt.plot(2025, predicted_2025, label='Prediction (2025)', marker='x', color='red')
plt.xlabel('Year')
plt.ylabel('Number of deaths')
plt.title('Prediction of the number of deaths from lung cancer (2020–2025)')
plt.legend()
plt.grid(True)
plt.show()
