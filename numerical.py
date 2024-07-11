import numpy as np
import pandas as pd
import glob

# 读取所有文件
height_files = sorted(glob.glob('2020to2024/height_month*.csv'))
salinity_files = sorted(glob.glob('2020to2024/salinity_month*.csv'))
temperature_files = sorted(glob.glob('2020to2024/temperature_month*.csv'))

height_data = np.array([pd.read_csv(file, header=None).values for file in height_files])
salinity_data = np.array([pd.read_csv(file, header=None).values for file in salinity_files])
temperature_data = np.array([pd.read_csv(file, header=None).values for file in temperature_files])

# 将数据整理为 (num_months, height, width) 的形状
height_data = np.array(height_data)
salinity_data = np.array(salinity_data)
temperature_data = np.array(temperature_data)

print(height_data.shape, salinity_data.shape, temperature_data.shape)

# %%
def fourier_series(x, n_terms):
    a0 = np.mean(x)
    terms = [a0]
    t = np.arange(len(x))
    for n in range(1, n_terms + 1):
        an = 2 / len(x) * np.sum(x * np.cos(2 * np.pi * n * t / len(x)))
        bn = 2 / len(x) * np.sum(x * np.sin(2 * np.pi * n * t / len(x)))
        terms.append(an)
        terms.append(bn)
    return terms

# 对每个点的时间序列进行傅里叶级数拟合
def fit_fourier_series(data, n_terms):
    num_months, height, width = data.shape
    coefficients = np.zeros((height, width, 2 * n_terms + 1))
    for i in range(height):
        for j in range(width):
            coefficients[i, j] = fourier_series(data[:, i, j], n_terms)
    return coefficients

n_terms = 5  # 傅里叶级数项数
height_coeffs = fit_fourier_series(height_data, n_terms)
salinity_coeffs = fit_fourier_series(salinity_data, n_terms)
temperature_coeffs = fit_fourier_series(temperature_data, n_terms)

print(height_coeffs.shape, salinity_coeffs.shape, temperature_coeffs.shape)


# %%
from sklearn.linear_model import LinearRegression

# 将傅里叶级数系数整理为特征矩阵
def prepare_features(*args):
    features = np.hstack([arg.reshape(-1, arg.shape[-1]) for arg in args])
    return features

# 准备特征矩阵和目标变量
X = prepare_features(salinity_coeffs, temperature_coeffs)
y = height_coeffs.reshape(-1, height_coeffs.shape[-1])

# 构建回归模型
model = LinearRegression()
model.fit(X, y)

print("Model coefficients:", model.coef_)

# %%
def predict_fourier_series(coeffs, n_terms, length):
    t = np.arange(length)
    a0 = coeffs[0]
    series = a0 / 2 * np.ones(length)
    for n in range(1, n_terms + 1):
        an = coeffs[2 * n - 1]
        bn = coeffs[2 * n]
        series += an * np.cos(2 * np.pi * n * t / length) + bn * np.sin(2 * np.pi * n * t / length)
    return series

# 预测未来一个月的傅里叶级数系数
num_months = height_data.shape[0]
X_pred = prepare_features(salinity_coeffs[-1:], temperature_coeffs[-1:])
y_pred_coeffs = model.predict(X_pred).reshape(height_coeffs.shape[0], height_coeffs.shape[1], -1)

# 用预测的傅里叶级数系数生成预测的高度数据
height_predicted = np.zeros((num_months + 1, height_data.shape[1], height_data.shape[2]))
height_predicted[:num_months] = height_data

for i in range(height_data.shape[1]):
    for j in range(height_data.shape[2]):
        height_predicted[num_months, i, j] = predict_fourier_series(y_pred_coeffs[i, j], n_terms, num_months + 1)

print("Predicted height data shape:", height_predicted.shape)

# 可视化预测结果
import matplotlib.pyplot as plt

# 选择一个特定的点进行可视化比较
i, j = 90, 90  # 中心点

plt.figure(figsize=(12, 6))
plt.plot(range(num_months), height_data[:, i, j], label='Original Data')
plt.plot(range(num_months + 1), height_predicted[:, i, j], label='Predicted Data', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Height')
plt.title('Original and Predicted Height Data')
plt.legend()
plt.show()
