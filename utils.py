import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from torchvision import models
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW,Adamax,RMSprop,Adadelta,Adagrad
import glob
import re
import os
import rasterio
from rasterio.transform import from_origin
from rasterio.transform import from_bounds
from rasterio.crs import CRS

# 自定义排序函数，用于提取文件名中的数字并转换为整数
def sort_key(filepath):
    # 只获取文件名，不包含路径
    filename = os.path.basename(filepath)
    # 假设文件名格式为 'prefix_number.csv'
    numbers = re.findall(r'\d+', filename)
    # 返回第一个找到的数字的整数形式，如果没有找到数字，则返回0
    return int(numbers[0]) if numbers else 0

def load_and_preprocess_data(_num_workers=4):
    # Reading all files
    file_prefix_test = '2021to2023'
    height_files_test = sorted(glob.glob(file_prefix_test + '/heightabovegeoid_*.csv'), key=sort_key)
    # pressure_files = sorted(glob.glob(file_prefix_test + '/pressureatseafloor_*.csv'))
    salinity_files_test = sorted(glob.glob(file_prefix_test + '/salinity_*.csv'), key=sort_key)
    temperature_files_test = sorted(glob.glob(file_prefix_test + '/temperature_*.csv'), key=sort_key)
    watermass_files_test = sorted(glob.glob(file_prefix_test + '/watermass_*.csv'), key=sort_key)

    file_prefix_train = '2001to2020'    
    height_files_train = sorted(glob.glob(file_prefix_train + '/height_*.csv'), key=sort_key)
    salinity_files_train = sorted(glob.glob(file_prefix_train + '/salinity_*.csv'), key=sort_key)
    temperature_files_train = sorted(glob.glob(file_prefix_train + '/temperature_*.csv'), key=sort_key)
    watermass_files_train = sorted(glob.glob(file_prefix_train + '/watermass_*.csv'), key=sort_key)

    height_data_test = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in height_files_test])
    # pressure_data = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in pressure_files_test])
    salinity_data_test = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in salinity_files_test])
    temperature_data_test = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in temperature_files_test])
    watermass_data_test = np.array([pd.read_csv(file, header=None).values for file in watermass_files_test])

    height_data_train = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in height_files_train])
    salinity_data_train = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in salinity_files_train])
    temperature_data_train = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in temperature_files_train])
    watermass_data_train = np.array([pd.read_csv(file, header=None).values for file in watermass_files_train])

    ## Training data
    num_samples = len(height_files_train)
    months = np.arange(num_samples)[:, np.newaxis,  np.newaxis]  # Add new axes for month
    months = np.tile(months, (240, 240))  # Repeat the month value to match spatial dimensions

    # Combine salinity, temperature, watermass, and month data as input features
    X = np.stack([salinity_data_train, temperature_data_train, watermass_data_train, months], axis=1)  # shape: (192, 4, 240, 240)
    y = height_data_train  # shape: (192, 240, 240)

    # Reshape data to (num_samples, num_channels, height, width)
    X = X.transpose(0, 2, 3, 1).reshape(-1, 4, 240, 240)  # shape: (192, 4, 240, 240)
    y = y.reshape(-1, 1, 240, 240)  # shape: (192, 1, 240, 240)

    # Standardize input features
    scaler = StandardScaler()
    X[:, :3, :, :] = scaler.fit_transform(X[:, :3, :, :].reshape(-1, X.shape[-1])).reshape(X[:, :3, :, :].shape) # 对四维数组X的前4个通道进行缩放处理
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    train_dataset=TensorDataset(X_tensor, y_tensor)

    ## Testing data
    num_samples_test = len(height_files_test)
    months_test = np.arange(num_samples_test)[:, np.newaxis,  np.newaxis]  # Add new axes for month
    months_test = np.tile(months_test+192, (240, 240))  # Repeat the month value to match spatial dimensions

    # Combine salinity, temperature, watermass, and month data as input features
    X_test = np.stack([salinity_data_test, temperature_data_test, watermass_data_test, months_test], axis=1)  # shape: (36, 4, 240, 240)
    y_test = height_data_test  # shape: (36, 240, 240)

    # Reshape data to (num_samples, num_channels, height, width)
    X_test = X_test.transpose(0, 2, 3, 1).reshape(-1, 4, 240, 240)  # shape: (36, 4, 240, 240)
    y_test = y_test.reshape(-1, 1, 240, 240)  # shape: (36, 1, 240, 240)

    # Standardize input features
    X_test[:, :3, :, :] = scaler.transform(X_test[:, :3, :, :].reshape(-1, X_test.shape[-1])).reshape(X_test[:, :3, :, :].shape)

    # Convert to tensors
    X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
    y_tensor_test = torch.tensor(y_test, dtype=torch.float32)
    test_dataset=TensorDataset(X_tensor_test, y_tensor_test)


    # # Create dataset and dataloaders
    # dataset = TensorDataset(X_tensor, y_tensor)
    # # train_size = int(0.8 * len(dataset))
    # # test_size = len(dataset) - train_size
    # # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # train_size = int(0.9 * len(dataset))
    # # train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

    # train_dataset=TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    # test_dataset=TensorDataset(X_tensor[train_size:], y_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=_num_workers,persistent_workers=True)
    val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=_num_workers,persistent_workers=True)
    # all_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=_num_workers,persistent_workers=True)
    
    return train_loader, val_loader

def load_and_preprocess_data_short(_num_workers=4):
    # Reading all files
    # file_prefix_test = '2021to2023'
    # height_files_test = sorted(glob.glob(file_prefix_test + '/heightabovegeoid_*.csv'), key=sort_key)
    # # pressure_files = sorted(glob.glob(file_prefix_test + '/pressureatseafloor_*.csv'))
    # salinity_files_test = sorted(glob.glob(file_prefix_test + '/salinity_*.csv'), key=sort_key)
    # temperature_files_test = sorted(glob.glob(file_prefix_test + '/temperature_*.csv'), key=sort_key)
    # watermass_files_test = sorted(glob.glob(file_prefix_test + '/watermass_*.csv'), key=sort_key)

    file_prefix_train = '2001to2020'    
    height_files_train = sorted(glob.glob(file_prefix_train + '/height_*.csv'), key=sort_key)
    salinity_files_train = sorted(glob.glob(file_prefix_train + '/salinity_*.csv'), key=sort_key)
    temperature_files_train = sorted(glob.glob(file_prefix_train + '/temperature_*.csv'), key=sort_key)
    watermass_files_train = sorted(glob.glob(file_prefix_train + '/watermass_*.csv'), key=sort_key)

    # height_data_test = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in height_files_test])
    # # pressure_data = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in pressure_files_test])
    # salinity_data_test = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in salinity_files_test])
    # temperature_data_test = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in temperature_files_test])
    # watermass_data_test = np.array([pd.read_csv(file, header=None).values for file in watermass_files_test])

    height_data_train = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in height_files_train])
    salinity_data_train = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in salinity_files_train])
    temperature_data_train = np.array([pd.read_csv(file, header=None, encoding="utf-8").values for file in temperature_files_train])
    watermass_data_train = np.array([pd.read_csv(file, header=None).values for file in watermass_files_train])

    ## Training data
    num_samples = len(height_files_train)
    months = np.arange(num_samples)[:, np.newaxis,  np.newaxis]  # Add new axes for month
    months = np.tile(months, (240, 240))  # Repeat the month value to match spatial dimensions

    # Combine salinity, temperature, watermass, and month data as input features
    X = np.stack([salinity_data_train, temperature_data_train, watermass_data_train, months], axis=1)  # shape: (192, 4, 240, 240)
    y = height_data_train  # shape: (192, 240, 240)

    # Reshape data to (num_samples, num_channels, height, width)
    X = X.transpose(0, 2, 3, 1).reshape(-1, 4, 240, 240)  # shape: (192, 4, 240, 240)
    y = y.reshape(-1, 1, 240, 240)  # shape: (192, 1, 240, 240)

    # Standardize input features
    scaler = StandardScaler()
    X[:, :3, :, :] = scaler.fit_transform(X[:, :3, :, :].reshape(-1, X.shape[-1])).reshape(X[:, :3, :, :].shape) # 对四维数组X的前4个通道进行缩放处理
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    # train_dataset=TensorDataset(X_tensor, y_tensor)

    # ## Testing data
    # num_samples_test = len(height_files_test)
    # months_test = np.arange(num_samples_test)[:, np.newaxis,  np.newaxis]  # Add new axes for month
    # months_test = np.tile(months_test+192, (240, 240))  # Repeat the month value to match spatial dimensions

    # # Combine salinity, temperature, watermass, and month data as input features
    # X_test = np.stack([salinity_data_test, temperature_data_test, watermass_data_test, months_test], axis=1)  # shape: (36, 4, 240, 240)
    # y_test = height_data_test  # shape: (36, 240, 240)

    # # Reshape data to (num_samples, num_channels, height, width)
    # X_test = X_test.transpose(0, 2, 3, 1).reshape(-1, 4, 240, 240)  # shape: (36, 4, 240, 240)
    # y_test = y_test.reshape(-1, 1, 240, 240)  # shape: (36, 1, 240, 240)

    # # Standardize input features
    # X_test[:, :3, :, :] = scaler.transform(X_test[:, :3, :, :].reshape(-1, X_test.shape[-1])).reshape(X_test[:, :3, :, :].shape)

    # # Convert to tensors
    # X_tensor_test = torch.tensor(X_test, dtype=torch.float32)
    # y_tensor_test = torch.tensor(y_test, dtype=torch.float32)
    # test_dataset=TensorDataset(X_tensor_test, y_tensor_test)


    # Create dataset and dataloaders
    dataset = TensorDataset(X_tensor, y_tensor)
    # # train_size = int(0.8 * len(dataset))
    # # test_size = len(dataset) - train_size
    # # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_size = int(0.8 * len(dataset))
    # train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]

    train_dataset=TensorDataset(X_tensor[:train_size], y_tensor[:train_size])
    test_dataset=TensorDataset(X_tensor[train_size:], y_tensor[train_size:])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=_num_workers,persistent_workers=True)
    val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=_num_workers,persistent_workers=True)
    
    return train_loader, val_loader


class SeaLevelResNet(pl.LightningModule):
    def __init__(self, input_channels):
        super(SeaLevelResNet, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改全连接层以适应输出大小
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 240 * 240)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1, 240, 240)  # 重塑为 (batch_size, 1, 240, 240)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # 将输出和标签展平，以便计算其他指标
        outputs_flat = outputs.view(-1).cpu().numpy()
        labels_flat = labels.view(-1).cpu().numpy()
        
        # 计算其他指标
        mse = mean_squared_error(labels_flat, outputs_flat)
        mae = mean_absolute_error(labels_flat, outputs_flat)
        r2 = r2_score(labels_flat, outputs_flat)
        
        # 记录指标
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', mse, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        self.log('val_r2', r2, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-4)
        return optimizer


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, dataloader, device):
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R^2 Score: {r2:.4f}')
    
    return mse, mae, r2

def save_as_tiff(data, filename, transform):
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data, 1)

def save_as_tiff(data, filename, transform, crs):
    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

def predict_future_sea_levels(model, num_month, device,write=True):
    model = model.to(device)
    model.eval()
    
    file_prefix = '2001to2020'
    file_postfix = str(num_month) + '.csv'
    height_data = np.array([pd.read_csv(file_prefix+'/height_' + file_postfix, header=None).values])
    # pressure_data = np.array([pd.read_csv(file_prefix+'/pressureatseafloor_' + file_postfix, header=None).values])
    salinity_data = np.array([pd.read_csv(file_prefix+'/salinity_' + file_postfix, header=None).values])
    temperature_data = np.array([pd.read_csv(file_prefix+'/temperature_' + file_postfix, header=None).values])
    watermass_data = np.array([pd.read_csv(file_prefix+'/watermass_' + file_postfix, header=None).values])

    month_data = np.full((1, 240, 240), num_month-1)

    # Combine salinity, temperature, watermass, and month data as input features
    X = np.stack([salinity_data, temperature_data, watermass_data, month_data], axis=1)  # shape: (1, 4, 240, 240)
    y = height_data  # shape: (1, 240, 240)

    # Reshape data to (num_samples, num_channels, height, width)
    X = X.transpose(0, 2, 3, 1).reshape(-1, 4, 240, 240)  # shape: (1, 4, 240, 240)

    # Standardize input features
    scaler = StandardScaler()
    X[:, :3, :, :] = scaler.fit_transform(X[:, :3, :, :].reshape(-1, X.shape[-1])).reshape(X[:, :3, :, :].shape)
    
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # Make prediction
    with torch.no_grad():
        predict_result = model(X_tensor)

    # Convert prediction results and true values to numpy arrays
    predict_result_np = np.rot90(predict_result.cpu().numpy().squeeze())
    y_np = np.rot90(y.squeeze())

    # Define affine transformation matrix and CRS
    left, bottom, right, top = -35, 40, -15, 60
    transform = from_bounds(left, bottom, right, top, predict_result_np.shape[1], predict_result_np.shape[0])
    crs = CRS.from_epsg(4326)

    diff = predict_result_np - y_np

    if write:
        # Save predicted and real values as TIFF files
        save_as_tiff(predict_result_np, f'predict/predicted_sea_levels_{num_month}.tiff', transform, crs)
        save_as_tiff(y_np, f'true/real_sea_levels_{num_month}.tiff', transform, crs)
        save_as_tiff(diff, f'predict/diff_sea_levels_{num_month}.tiff', transform, crs)

    return predict_result.cpu().numpy().squeeze(), y.squeeze()

def predict_all_future_sea_levels(model, device, month_extent=[132,162]):
    all_predictions = []
    all_true_values = []
    for i in range(month_extent[0], month_extent[1]+1):
        prediction, true_values = predict_future_sea_levels(model, i, device,True)
        # 将每个预测结果平坦化成一维数组
        all_predictions.append(prediction.flatten())
        all_true_values.append(true_values.flatten())

        # 将预测结果转换为 DataFrame 并保存
        df_pred = pd.DataFrame(prediction)
        filename_pred = f'predict/height_predictions_{i}.csv'
        df_pred.to_csv(filename_pred, index=False)

        # 假设你也想保存真实值
        df_true = pd.DataFrame(true_values)
        filename_true = f'true/height_true_values_{i}.csv'
        df_true.to_csv(filename_true, index=False)

    # 将列表转换为二维数组
    all_predictions_2d = np.array(all_predictions)
    all_true_values_2d = np.array(all_true_values)

    # 计算所有预测值和真实值的平均值
    avg_predictions = pd.DataFrame(all_predictions_2d).mean(axis=1)
    avg_true_values = pd.DataFrame(all_true_values_2d).mean(axis=1)

    # 保存平均预测值和平均真实值到 CSV 文件
    avg_predictions.to_csv('predict/average_predictions.csv', index=False)
    avg_true_values.to_csv('true/average_true_values.csv', index=False)

    return all_predictions

def save_time_series_data_for_point(point, output_file):
    # 读取所有文件5
    file_prefix = '2021to2023'
    height_files = sorted(glob.glob(file_prefix+'/heightabovegeoid_*.csv'))
    # pressure_files = sorted(glob.glob(file_prefix+'/pressureatseafloor_*.csv'))
    salinity_files = sorted(glob.glob(file_prefix+'/salinity_*.csv'))
    temperature_files = sorted(glob.glob(file_prefix+'/temperature_*.csv'))
    watermass_files = sorted(glob.glob(file_prefix+'/watermass_*.csv'))
    
    i, j = point # i 为经度，j 为纬度
    all_data = []

    # 遍历所有文件，确保时间顺序
    for height_file, pressure_file, salinity_file, temperature_file, watermass_file in zip(height_files, salinity_files, temperature_files, watermass_files):
        height_data = pd.read_csv(height_file, header=None).values
        # pressure_data = pd.read_csv(pressure_file, header=None).values
        salinity_data = pd.read_csv(salinity_file, header=None).values
        temperature_data = pd.read_csv(temperature_file, header=None).values
        watermass_data = pd.read_csv(watermass_file, header=None).values
        
        # pressure= pressure_data[i, j]
        salinity = salinity_data[i, j]
        temperature = temperature_data[i, j]
        watermass = watermass_data[i, j]
        height = height_data[i, j]
        
        all_data.append([salinity, temperature, watermass, height])

    # 转换为 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(all_data, columns=['Salinity', 'Temperature', 'Watermass', 'Height'])
    df.to_csv(output_file, index=False)


