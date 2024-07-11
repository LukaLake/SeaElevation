from utils import *
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    ifTrained = True
    torch.set_float32_matmul_precision('high')

    # 假设你的数据已经加载和预处理好
    train_loader, val_loader = load_and_preprocess_data_short(4)
    
    # 获取输入数据的形状以初始化模型
    sample_input, _ = next(iter(train_loader))
    _, input_channels, height, width = sample_input.shape

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='lightning_logs/checkpoints',
        filename='model-{epoch:02d}-{val_r2:.4f}',
        save_top_k=1,
        mode='min'
    )

    # 初始化模型
    model = SeaLevelResNet(input_channels)
    
    if (ifTrained):
        # 预测未来一个月的高度数据
        # 载入最优的模型检查点
        # best_model_path = checkpoint_callback.best_model_path
        best_model_path = 'best_model/Adam_0.0001_resnet101_model-epoch=31-val_r2=0.89.pth'
        print(f"Best model saved at: {best_model_path}")

        # 使用最佳模型进行评估
        # model = SeaLevelResNet.load_from_checkpoint(best_model_path)
        # checkpoint = torch.load(best_model_path)
        # model.load_state_dict(checkpoint['state_dict'])
        model= torch.load(best_model_path)
        evaluate_model(model, val_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        predict_future_sea_levels(model, 160, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        predict_all_future_sea_levels(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # torch.save(model, f'best_model/model-epoch=40-val_r2=0.90.pth')
        # print(f"Best model saved at: best_model/model-epoch=40-val_r2=0.90.pth")

        # save_time_series_data_for_point((10, 10), 'time_series_data_10_10.csv')
        # save_time_series_data_for_point((170, 10), 'time_series_data_170_10.csv')
        
    else:
        # 定义训练器并添加回调
        trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=400)
        
        # 训练模型
        trainer.fit(model, train_loader, val_loader)

        best_model_path = checkpoint_callback.best_model_path
        # 获取文件名和扩展名
        filename_with_extension = os.path.basename(best_model_path)

        # 分割文件名和扩展名
        filename_without_extension, _ = os.path.splitext(filename_with_extension)
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])

        evaluate_model(model, val_loader, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model_name = f'AdamW_0.0001_resnet101_{filename_without_extension}.pth'
        torch.save(model, f'best_model/{model_name}')
        print(f"Best model saved at: best_model/{model_name}")




