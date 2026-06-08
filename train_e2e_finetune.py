# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from model.yinglong_st.afno_attention_parallel_station_grad import AFNOAttnParallelNet
from model.yinglong_st.afno_attention_parallel import AFNOAttnParallelNet as YLNet
from model.cat_da import CatDAModel_eval
from dataset.da_dataset_4dvar_grad_DFD import make_splits
# from loss.l2 import RelativeL2Loss
from loss.l2 import FocalFrequencyLoss
import logging


def train_model(current_epoch, epochs, model, criterion, optimizer, dataloader, scheduler, accelerator):
    
    adj = torch.tensor(np.load('data_stats/ED_stat/position.npy')).to(accelerator.device)
    model.train()

    
    for epoch in range(current_epoch, epochs):
        total_loss = torch.tensor(0.0).to(accelerator.device)

        for i, batch in enumerate(dataloader):

            inputs, targets, goes_data, station_data, time_list, geo = batch

            optimizer.zero_grad()
            outputs, bg, outgrad = model(inputs, goes_data, station_data, time_list, geo, adj)
            l2loss, freloss, gradloss, stloss = criterion(bg, outputs, targets, outgrad, station_data[:,:,:,:4])
            loss = l2loss + freloss + gradloss + stloss
            accelerator.backward(loss)
            
            accelerator.clip_grad_norm_(model.parameters(), max_norm = 1.0)
            
            optimizer.step()

            total_loss += loss.item()

            outputs = torch.stack(outputs, dim=1)

            if (i + 1) % 1 == 0:
                logging.info(f"{accelerator.device} Train... [epoch {epoch + 1}/{epochs}, step {i + 1}/{len(dataloader)}]\t[L2loss {l2loss.item()}]\t[Specloss {freloss.item()}]\t[Gradloss {gradloss.item()}]\t[Stloss {stloss.item()}]")

        avg_loss = total_loss / len(dataloader)      

        accelerator.wait_for_everyone()
        reduced_avg_loss = accelerator.reduce(avg_loss, "mean")  

        scheduler.step()
        if accelerator.is_main_process:
            logging.info(f"Epoch {epoch + 1}, Epoch Avg Loss: {reduced_avg_loss.item()}, L_R: {optimizer.param_groups[0]['lr']}")
            save_model = accelerator.unwrap_model(model)
            save_path = f"/nfs/samba/数据聚变/气象数据/DA_data/output/yinglong_ED/e2e/model_da_epoch_{epoch + 1}.pth"
            # 将需要保存的信息打包成一个字典
            checkpoint = {
                'epoch': epoch+1,  # 当前的 epoch
                'model_state_dict': save_model.state_dict(),  # 模型的权重
                'optimizer_state_dict': optimizer.state_dict(),  # 优化器的状态
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,  # 调度器的状态
                'loss': loss,  # 当前的损失值
                'learning_rate': optimizer.param_groups[0]['lr']  # 当前的学习率
            }

            # 保存到文件
            torch.save(checkpoint, save_path)
            # torch.save(save_model.state_dict(), save_path)
            print(f"Model parameters saved to {save_path}")

def main():

    logging.basicConfig(
    filename='log/yl+G_e2e/log_output_e2e_fine  .txt',        # 指定文件名
    filemode='a',                     # 文件追加模式
    level=logging.INFO,               # 日志级别，INFO 级别以上的日志会被记录
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式，包含时间、日志级别和消息
    )

    accelerator = Accelerator()
    print(f"Using device: {accelerator.device}")

    current_epoch = 0
    epochs = 30
    batch_size = 4

    IMG_H, IMG_W = 440, 408
    in_channels, out_channels = 37, 24
    num_timestamps = 4
    da_lead_time = 1
    stride = 1
    yl_path = '/nfs/samba/数据聚变/气象数据/DA_data/output/yinglong_ED/grad_fine_epoch_15.pth'
    
    # 创建模型、优化器和数据
    DAmodel = AFNOAttnParallelNet(
        img_size=(IMG_H, IMG_W),
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=768,
        num_timestamps=num_timestamps,
        attn_channel_ratio=0.25
    )
    
    YLmodel = YLNet(
        img_size=(IMG_H, IMG_W),
        in_channels=25,
        out_channels=24,
        num_timestamps=num_timestamps,
        attn_channel_ratio=0.25,
        cnn_channel_ratio=0.25
    )

    model = CatDAModel_eval(DAmodel, yl_path, YLmodel, num_timestamps)
    local_indices = np.load('data_stats/ED_stat/local.npy')
    criterion = FocalFrequencyLoss(local_indices,loss_weight=2)

    optimizer = torch.optim.Adam(model.parameters(), lr= 5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
	    T_max = epochs , 
	    eta_min=0, 
	    last_epoch=- 1)

    load = True
    if load :
        # 加载保存的 checkpoint
        checkpoint = torch.load('/nfs/samba/数据聚变/气象数据/DA_data/output/yinglong_ED/e2e/model_da_epoch_30.pth', weights_only = True)
        intermediate_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            if key.startswith('intermediate.'):
                new_key = key.replace('intermediate.', '')
                intermediate_state_dict[new_key] = value
        model.intermediate.load_state_dict(intermediate_state_dict)
        # model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # current_epoch = checkpoint['epoch']
    # 准备模型和优化器
    train_loader, _ = make_splits(
            data_path = "/nfs/samba/数据聚变/气象数据/DA_data/440_ED_data/hrrr_rb.zarr",
            background_data_path = "/nfs/samba/数据聚变/气象数据/DA_data/440_ED_data/era5_rb.zarr",
            goes_data_path= "/nfs/samba/数据聚变/气象数据/DA_data/440_ED_data/goes_rb.zarr",
            station_data_path="/nfs/samba/数据聚变/气象数据/DA_data/440_ED_data/hadisd_rb.nc",
            stat_path =  "data_stats/ED_stat/data_stats.csv",
            batch_size=batch_size,
            var_list = {'z': [('50', 'z50'), ('500', 'z500'), ('850', 'z850'), ('1000', 'z1000')],
                        't': [('50', 't50'), ('500', 't500'), ('850', 't850'), ('1000', 't1000')],
                        's': [('50', 's50'), ('500', 's500'), ('850', 's850'), ('1000', 's1000')],
                        'u': [('50', 'u50'), ('500', 'u500'), ('850', 'u850'), ('1000', 'u1000')],
                        'v': [('50', 'v50'), ('500', 'v500'), ('850', 'v850'), ('1000', 'v1000')],
                        'mslp': [('surface', 'mslp')],
                        'u10': [('surface', 'u10')],
                        'v10': [('surface', 'v10')],
                        't2m': [('surface', 't2m')]
                        },
            goes_band = ['02','07','08','09','10','14'], 
            station_var = ['mslp','u10','v10','t2m'], 
            train_time_range = ['2019-1-1', '2023-12-31'],
            valid_time_range = ['2024-1-1', '2024-1-2'],

            stride = stride,
            num_timestamps = num_timestamps,
            lead_time = da_lead_time,
            data_random = True)
        

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        # for idx, (name, param) in enumerate(model.named_parameters()):
        #     print(idx, name)

    train_model(current_epoch, epochs, model, criterion, optimizer, train_loader, scheduler, accelerator)

    accelerator.end_training()

if __name__ == "__main__":
    main()