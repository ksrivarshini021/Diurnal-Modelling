from collections import defaultdict
import torch
# from preTrainedTemporalSpatial3 import maeViT
# from preTrainedmaeViT import maeViT
# from UnetBaseline import UNet
# from TStransformer import STTransformerBaseline
# from TimeSformerArch import TimeSformerBaseline
# from BiLSTMArch import BiLSTM
from preTrainedTemporal3 import maeViT
from vitTrainLongST3 import train_model, SatelliteImageDataset, test_model, plot_lst_heatmaps
import os
import natsort
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, Dataset
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
import rasterio
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim 
from torchmetrics.image.fid import FrechetInceptionDistance as FID
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
from datetime import datetime, timedelta
from collections import defaultdict
import timm
import lpips
lpips_fn = lpips.LPIPS(net='alex')  
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fid_metric = FID(feature=2048, normalize=True).to(device)
lpips_fn = lpips.LPIPS(net='alex').to(device)

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = maeViT()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    print(f"Model is in {'evaluation' if model.training == False else 'training'} mode.")


    train_losses = checkpoint.get("train_losses", [])
    test_losses = checkpoint.get("test_losses", []) 

    if train_losses and test_losses:
        last_epoch = checkpoint.get("epoch", len(train_losses) - 1)
        print(f"Checkpoint saved at epoch: {last_epoch}")
        print(f"Last Train Loss for epoch {last_epoch}: {train_losses[-1]:.4f}")
        print(f"Last Test Loss for epoch{last_epoch}: {test_losses[-1]:.4f}")
    else:
        print("Loss histories not found in checkpoint.")

    return model, checkpoint.get('epoch', 0)

def prepare_test_data():
    input_dir = "/s/chopin/e/proj/hyperspec/diurnalModel/combinedQuadHash2"
    target_dir = "/s/chopin/e/proj/hyperspec/diurnalModel/LSTCTargetTest"

    valid_days = sorted(os.listdir(input_dir))
    # valid_days = valid_days[:3]

    dataset = SatelliteImageDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        valid_days=valid_days,
    )

    split_idx = int(0.8 * len(dataset))
    test_indices = list(range(split_idx, len(dataset)))  
    test_dataset = Subset(dataset, test_indices)        

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        drop_last=True,
        num_workers=4,
    )

    print(f"Total samples: {len(dataset)} | Test samples: {len(test_dataset)}")
    return test_dataloader

def plot_lst_heatmaps_test(lst_data, is_target=False, epoch=None, day=None, hour=None, quad_hash=None):
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    fig.suptitle(f'Land Surface Temperature (24 Hours)', fontsize=16)

    lst_data = np.array(lst_data)  

    # lst_data = lst_data.detach().numpy()
    if lst_data.size == 0:
        print(f"Warning: lst_data is empty for epoch {epoch}, batch {batch_idx}")
        return  
    
    # print(f"lst_data shape: {lst_data.shape}, is_target: {is_target}") 
    masked_data = np.ma.masked_where(lst_data < 0, lst_data) 

    for hour in range(24):
        ax = axes[hour // 6, hour % 6]
        cax = ax.imshow(masked_data[hour], cmap='hot')
        ax.set_title(f'Hour {hour:02d}')
        ax.axis('off')

    fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.05, label='Temperature')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if is_target:
        plt.savefig(f'maeVitResultsTemporal+Spatial/test_target_{day}_{hour}_{quad_hash}_epoch{epoch+1}.png')
    else:
        plt.savefig(f'maeVitResultsTemporal+Spatial/test_predicted_{day}_{hour}_{quad_hash}_epoch{epoch+1}.png')
    plt.close()

def hour_psnr(model, dataloader, device):
    model.eval()
    psnr_values = defaultdict(list)
    with torch.no_grad():
        for batch_idx, (inputs, targets, temporal_info, spatial_info) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            temporal_info = temporal_info.to(device)
            # spatial_info = spatial_info.to(device)
            outputs = model(inputs, temporal_info)
            # outputs = model(inputs)

            for i in range(inputs.size(0)):
                for hour in range(24):
                    target_hour = targets[i, hour]
                    output_hour = outputs[i, hour]

                    valid_mask = target_hour != 0
                    if valid_mask.sum() == 0:
                        continue

                    mse = F.mse_loss(output_hour[valid_mask], target_hour[valid_mask])
                    psnr = 20 * torch.log10(torch.tensor(1.0).to(device)) - 10 * torch.log10(mse)
                    psnr_values[hour].append(psnr.item())

    avg_psnr_hr = []
    for hour in range(24):
        hour_psnrs = psnr_values.get(hour, [])
        if hour_psnrs:
            avg_psnr = sum(hour_psnrs) / len(hour_psnrs)
        else:
            avg_psnr = float('nan')
        avg_psnr_hr.append(avg_psnr)
    return avg_psnr_hr


def extract_month(filename):
    parts = filename.split('_')
    if len(parts) != 3:
        print(f"Unexpected filename format: {filename}")
        return None

    day_str = parts[0] 
    try:
        day_of_year = int(day_str)

        fake_date = datetime(2022, 1, 1) + timedelta(days=day_of_year - 1)

        month = fake_date.month
        return month
    except Exception as e:
        print(f"Error processing day format for {filename}: {e}")
        return None

def extract_hour(filename):
    parts = filename.split('_')
    if len(parts) != 3:
        print(f"Unexpected filename format: {filename}")
        return None

    hour_str = parts[1]
    try:
        hour = int(hour_str)
        return hour
    except Exception as e:
        print(f"Error processing hour format for {filename}: {e}")
        return None


def month_psnr(model, dataloader, device):
    model.eval()
    psnr_by_month = defaultdict(list)

    original_dataset = dataloader.dataset.dataset
    indices = dataloader.dataset.indices

    with torch.no_grad():
        for batch_idx, (inputs, targets, temporal_info, spatial_info) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            temporal_info = temporal_info.to(device)
            # spatial_info = spatial_info.to(device)
            outputs = model(inputs, temporal_info)
            # outputs = model(inputs)

            for i in range(inputs.size(0)):
                original_index = indices[batch_idx * dataloader.batch_size + i]
                input_path = original_dataset.input_files[original_index]
                filename = os.path.basename(input_path)

                parts = filename.split('_')
                if len(parts) != 3:
                    print(f"Unexpected filename format: {filename}")
                    continue

                day_str = parts[0] 
                try:
                    day_of_year = int(day_str)
                    fake_date = datetime(2020, 1, 1) + timedelta(days=day_of_year - 1)
                    month = fake_date.month
                except Exception as e:
                    print(f"Could not parse day-of-year from filename: {filename} — {e}")
                    continue

                psnrs = []
                for hour in range(24):
                    target_hour = targets[i, hour]
                    output_hour = outputs[i, hour]
                    valid_mask = target_hour != 0
                    if valid_mask.sum() == 0:
                        continue
                    mse = F.mse_loss(output_hour[valid_mask], target_hour[valid_mask])
                    psnr = 20 * torch.log10(torch.tensor(1.0).to(device)) - 10 * torch.log10(mse)
                    psnrs.append(psnr.item())

                if psnrs:
                    psnr_avg = sum(psnrs) / len(psnrs)
                    psnr_by_month[month].append(psnr_avg)

    avg_psnr_month = []
    for m in range(1, 13):
        month_scores = psnr_by_month.get(m, [])
        if month_scores:
            avg = sum(month_scores) / len(month_scores)
        else:
            avg = float('nan')
        avg_psnr_month.append(avg)

    return avg_psnr_month



def calculate_psnr(outputs, targets, device, mask_value=-1, max_pixel_value=1.0):
    psnrs = []
    for hour in range(24):
        target_hour = targets[hour].to(device)
        output_hour = outputs[hour].to(device)

        mask = (target_hour != mask_value).float()
        valid_pred = output_hour * mask
        valid_target = target_hour * mask

        mse = torch.mean((valid_pred - valid_target) ** 2)
        mse = torch.clamp(mse, 1e-8)  

        psnr = 20 * torch.log10(
            torch.tensor(max_pixel_value, device=target_hour.device)
        ) - 10 * torch.log10(mse)

        psnrs.append(psnr.item())

    return sum(psnrs) / len(psnrs) if psnrs else float('nan')


def calculate_ssim(outputs, targets, device, mask_value = -1):
    ssim_values = []
    for hour in range(24):
        target_hour = targets[hour].to(device)
        output_hour = outputs[hour].to(device)
        if target_hour.dim() == 2:
            target_hour = target_hour.unsqueeze(0).unsqueeze(0)  
            output_hour = output_hour.unsqueeze(0).unsqueeze(0)

        elif target_hour.dim() == 3:
            target_hour = target_hour.unsqueeze(0) 
            output_hour = output_hour.unsqueeze(0)

        ssim_value = ssim(output_hour, target_hour, data_range=1.0, reduction='elementwise_mean')
        ssim_values.append(ssim_value.item())

    return sum(ssim_values) / len(ssim_values) if ssim_values else float('nan')


def calculate_lpips(outputs, targets, device, mask_value=-1):
    lpips_values = []
    for hour in range(24):
        target_hour = targets[hour].to(device)
        output_hour = outputs[hour].to(device)
        # print(f"Target hour shape: {target_hour.shape}, Output hour shape: {output_hour.shape}")

        if target_hour.dim() == 2:
            target_hour = target_hour.unsqueeze(0)  
            output_hour = output_hour.unsqueeze(0)
            target_input = target_hour.repeat(3, 1, 1).unsqueeze(0)  
            output_input = output_hour.repeat(3, 1, 1).unsqueeze(0)
        elif target_hour.dim() == 3:
            bands = target_hour.shape[0]
            if bands == 3:
                target_input = target_hour.unsqueeze(0)  
                output_input = output_hour.unsqueeze(0)
            else:
                target_gray = target_hour.mean(dim=0, keepdim=True).unsqueeze(0)  
                output_gray = output_hour.mean(dim=0, keepdim=True).unsqueeze(0)
                target_input = target_gray.repeat(1, 3, 1, 1)
                output_input = output_gray.repeat(1, 3, 1, 1)
        elif target_hour.dim() == 4:
            bands = target_hour.shape[1]
            if bands == 3:
                target_input = target_hour
                output_input = output_hour
            else:
                target_gray = target_hour.mean(dim=1, keepdim=True)
                output_gray = output_hour.mean(dim=1, keepdim=True)
                target_input = target_gray.repeat(1, 3, 1, 1)
                output_input = output_gray.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Unexpected target tensor dimension: {target_hour.shape}")
        target_input = (target_input * 2) - 1
        output_input = (output_input * 2) - 1

        score = lpips_fn(output_input, target_input)
        lpips_values.append(score.item())

    return sum(lpips_values) / len(lpips_values) if lpips_values else float('nan')


def calculate_CC(outputs, targets, device, mask_value=-1):
    cc_values = []
    for hour in range(24):
        target_hour = targets[hour].to(device)
        output_hour = outputs[hour].to(device)
        mask = (target_hour != mask_value)
        target_valid = target_hour[mask]
        output_valid = output_hour[mask]

        if target_valid.numel() == 0:
            continue

        target_centered = target_valid - target_valid.mean()
        output_centered = output_valid - output_valid.mean()

        numerator = (target_centered * output_centered).sum()
        denominator = torch.sqrt((target_centered ** 2).sum() * (output_centered ** 2).sum())
        cc = numerator / denominator if denominator != 0 else torch.tensor(0.0, device=device)
        cc_values.append(cc.item())

    return sum(cc_values) / len(cc_values) if cc_values else float('nan')


def calculate_ergas(outputs, targets, device, ratio=1, mask_value=-1):
    ergas_values = []
    for hour in range(24):
        target_hour = targets[hour].to(device)
        output_hour = outputs[hour].to(device)

        mask = (target_hour != mask_value)

        bands = target_hour.shape[0]
        rmse_list, mean_list = [], []

        for b in range(bands):
            band_mask = mask[b]
            if band_mask.sum() == 0:
                continue

            target_valid = target_hour[b][band_mask]
            output_valid = output_hour[b][band_mask]

            rmse = torch.sqrt(torch.mean((target_valid - output_valid) ** 2))
            mean_val = target_valid.mean()

            rmse_list.append(rmse)
            mean_list.append(mean_val)

        if rmse_list:
            rmse_arr = torch.stack(rmse_list)
            mean_arr = torch.stack(mean_list)
            ergas = 100 / ratio * torch.sqrt(torch.sum((rmse_arr / mean_arr) ** 2) / bands)
            ergas_values.append(ergas.item())

    return sum(ergas_values) / len(ergas_values) if ergas_values else float('nan')


def calculate_mae(outputs, targets, device, mask_value=-1):
    """
    Compute Mean Absolute Error (MAE) over 24 hours with optional masking.
    """
    mae_values = []  # avoid name conflict
    for hour in range(24):
        target_hour = targets[hour].to(device)
        output_hour = outputs[hour].to(device)

        mask = (target_hour != mask_value).float()
        valid_pred = output_hour * mask
        valid_target = target_hour * mask

        if mask.sum() == 0:
            continue

        mae_val = torch.sum(torch.abs(valid_pred - valid_target)) / mask.sum()
        mae_values.append(mae_val.item())

    return sum(mae_values) / len(mae_values) if mae_values else float('nan')



def calculate_rmse(outputs, targets, device, mask_value=-1):
    """
    Compute Root Mean Squared Error (RMSE) over 24 hours with optional masking.
    """
    rmse_values = []
    for hour in range(24):
        target_hour = targets[hour].to(device)
        output_hour = outputs[hour].to(device)

        mask = (target_hour != mask_value).float()
        valid_pred = output_hour * mask
        valid_target = target_hour * mask

        if mask.sum() == 0:
            continue

        mse = torch.sum((valid_pred - valid_target) ** 2) / mask.sum()
        rmse_val = torch.sqrt(mse)
        rmse_values.append(rmse_val.item())

    return sum(rmse_values) / len(rmse_values) if rmse_values else float('nan')


# def calculate_fid(outputs, targets, device, mask_value=-1):
#     fid_metric.reset()
#     for hour in range(24):
#         real = targets[hour].to(device)
#         fake = outputs[hour].to(device)

#         mask = (real != mask_value).float()
#         real = real * mask
#         fake = fake * mask

#         real = real.unsqueeze(0)
#         fake = fake.unsqueeze(0)

#         fid_metric.update(fake, real)

#     return fid_metric.compute().item()

def evaluate_model(model, dataloader, device, mask_value=-1, max_pixel_value=1.0):
    model.eval()
    psnr_scores, rmse_scores, mae_scores, ssim_scores, lpips_scores, cc_scores, ergas_scores = [], [], [], [], [], [], []

    with torch.no_grad():
        for inputs, targets, spatial_info, temporal_info in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            temporal_info = temporal_info.to(device)
            # spatial_info = spatial_info.to(device)

            outputs = model(inputs, temporal_info)
            # outputs = model(inputs)

            for i in range(inputs.size(0)): 
                # psnr = calculate_psnr(outputs[i], targets[i], device, mask_value, max_pixel_value)
                # rmse = calculate_rmse(outputs[i], targets[i], device, mask_value)
                mae  = calculate_mae(outputs[i], targets[i], device, mask_value)
                # ssim_val = calculate_ssim(outputs[i], targets[i], device, mask_value)
                # lpips_val = calculate_lpips(outputs[i], targets[i], device=device, mask_value=mask_value)
                # cc_val = calculate_CC(outputs[i], targets[i], device=device, mask_value=mask_value)
                # ergas_val = calculate_ergas(outputs[i], targets[i], device=device, mask_value=mask_value)

                # lpips_scores.append(lpips_val)
                # cc_scores.append(cc_val)
                # psnr_scores.append(psnr)
                # rmse_scores.append(rmse)
                mae_scores.append(mae)
                # ssim_scores.append(ssim_val)
                # ergas_scores.append(ergas_val)

    return {
        # 'PSNR': sum(psnr_scores) / len(psnr_scores) if psnr_scores else float('nan'),
        # 'RMSE': sum(rmse_scores) / len(rmse_scores) if rmse_scores else float('nan'),
        'MAE':  sum(mae_scores) / len(mae_scores) if mae_scores else float('nan'),
        # 'SSIM': sum(ssim_scores) / len(ssim_scores) if ssim_scores else float('nan'),
        # 'LPIPS': sum(lpips_scores) / len(lpips_scores) if lpips_scores else float('nan'),
        # 'CC':    sum(cc_scores) / len(cc_scores) if cc_scores else float('nan'),
        # 'ERGAS': sum(ergas_scores) / len(ergas_scores) if ergas_scores else float('nan'),
    }





# def monthly_hourly_psnr(model, dataloader, device):
#     model.eval()
#     psnr_by_month_hour = defaultdict(lambda: defaultdict(list))

#     original_dataset = dataloader.dataset.dataset
#     indices = dataloader.dataset.indices

#     with torch.no_grad():
#         for batch_idx, (inputs, targets, temporal_info) in enumerate(dataloader):
#             inputs = inputs.to(device)
#             targets = targets.to(device)
#             temporal_info = temporal_info.to(device)
#             outputs = model(inputs, temporal_info)

#             for i in range(inputs.size(0)):
#                 original_index = indices[batch_idx * dataloader.batch_size + i]
#                 input_path = original_dataset.input_files[original_index]
#                 filename = os.path.basename(input_path)

#                 month = extract_month(filename)
#                 if month is None:
#                     continue 

#                 psnrs = calculate_psnr(outputs[i], targets[i], device)

#                 for hour in range(24):
#                     psnr_by_month_hour[month][hour].append(psnrs[hour])

#     avg_psnr_by_month_hour = defaultdict(lambda: defaultdict(list))
#     for month, hour_dict in psnr_by_month_hour.items():
#         for hour, psnrs in hour_dict.items():
#             avg_psnr_by_month_hour[month][hour] = sum(psnrs) / len(psnrs)

#     fig, axes = plt.subplots(3, 4, figsize=(18, 12))  

#     months = list(range(1, 13))
#     for month_idx, month in enumerate(months):
#         ax = axes[month_idx // 4, month_idx % 4]
#         avg_psnr_per_hour = [avg_psnr_by_month_hour[month].get(hour, float('nan')) for hour in range(24)]
#         ax.plot(range(24), avg_psnr_per_hour, marker='o', label=f'Month {month}')
#         ax.set_title(f'Month {month}')
#         ax.set_xlabel('Hour of the day')
#         ax.set_ylabel('Avg PSNR')
#         ax.set_xticks(range(0, 24, 2)) 
#         ax.grid(True)
#         ax.legend()

#     plt.tight_layout()
#     plt.savefig('maeVitResultsTemporal/monthly_hourly_psnr_chart.png')
#     plt.show()

#     return avg_psnr_by_month_hour

def monthly_hourly_psnr(model, dataloader, device):

    def extract(filename):
        month = extract_month(filename)
        hour = extract_hour(filename)
        if month is None or hour is None:
            return None, None
        return month, hour
    
    def cal_psnr(pred, target):
        valid_mask = target != 0
        if valid_mask.sum() == 0:
            return None
        mse = F.mse_loss(pred[valid_mask], target[valid_mask])
        psnr = 20 * torch.log10(torch.tensor(1.0).to(device)) - 10 * torch.log10(mse)
        return psnr.item()
    
    model.eval()
    psnr_outputs = defaultdict(lambda: defaultdict(list))

    original_dataset = dataloader.dataset.dataset
    indices = dataloader.dataset.indices

    with torch.no_grad():
        for batch_idx, (inputs, targets, temporal_info, spatial_info) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            temporal_info = temporal_info.to(device)
            # spatial_info = spatial_info.to(device)
            outputs = model(inputs, temporal_info)
            # outputs = model(inputs)

            for i in range(inputs.size(0)):
                original_index = indices[batch_idx * dataloader.batch_size + i]
                filename = os.path.basename(original_dataset.input_files[original_index])
                month, hour = extract(filename)
                if month is None or hour is None:
                    continue

                for h in range(24):
                    output_psnr = cal_psnr(outputs[i, h], targets[i, h])
                    if output_psnr is not None:
                        psnr_outputs[month][h].append(output_psnr)


    avg_output_psnr = np.full((12, 24), np.nan)

    for month in range(1, 13):
        for hour in range(24):
            output_values = psnr_outputs[month][hour]
            if output_values:
                avg_output_psnr[month - 1, hour] = np.mean(output_values)


    fig, axes = plt.subplots(3, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    hours = np.arange(24)

    for m in range(12):
        ax = axes[m]
        ax.set_title(f'Month {m + 1}')
        ax.set_xticks(hours[::4])
        ax.set_ylim(0, 50)
        ax.plot(hours, avg_output_psnr[m], label='', marker='x')
        ax.grid(True)
        if m % 3 == 0:
            ax.set_ylabel("PSNR")
        if m >= 9:
            ax.set_xlabel("Hour")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=12)
    fig.suptitle("Hourly PSNR per Month", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("maeVitResultsTemporal+Spatial/monthly_hourly_psnr_lineplots.png")
    plt.show()

    return avg_output_psnr

                

                
    # psnr_by_month_hour = defaultdict(lambda: defaultdict(list))

    # original_dataset = dataloader.dataset.dataset
    # indices = dataloader.dataset.indices

    # with torch.no_grad():
    #     for batch_idx, (inputs, targets, temporal_info, spatial_info) in enumerate(dataloader):
    #         inputs = inputs.to(device)
    #         targets = targets.to(device)
    #         temporal_info = temporal_info.to(device)
    #         spatial_info = spatial_info.to(device)
    #         outputs = model(inputs, temporal_info, spatial_info)

    #         for i in range(inputs.size(0)):
    #             original_index = indices[batch_idx * dataloader.batch_size + i]
    #             input_path = original_dataset.input_files[original_index]
    #             filename = os.path.basename(input_path)

    #             month = extract_month(filename)
    #             hour = extract_hour(filename)
    #             if month is None or hour is None:
    #                 continue

    #             psnrs = calculate_psnr(outputs[i], targets[i], device)

    #             if isinstance(psnrs, list) or isinstance(psnrs, np.ndarray):
    #                 if 0 <= hour < len(psnrs):
    #                     psnr_by_month_hour[month][hour].append(psnrs[hour])
    #             else:
    #                 psnr_by_month_hour[month][hour].append(psnrs)

    # heatmap_data = np.full((12, 24), np.nan)
    # for month in range(1, 13):
    #     for hour in range(24):
    #         psnrs = psnr_by_month_hour[month][hour]
    #         if psnrs:
    #             heatmap_data[month - 1, hour] = sum(psnrs) / len(psnrs)

    # plt.figure(figsize=(14, 6))
    # im = plt.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower')
    # plt.colorbar(im, label='Average PSNR')
    # plt.title('Average PSNR by Hour and Month')
    # plt.xlabel('Hour of Day')
    # plt.ylabel('Month')
    # plt.xticks(np.arange(0, 24, 1))
    # plt.yticks(np.arange(12), [f'Month {i+1}' for i in range(12)])
    # plt.grid(False)
    # plt.tight_layout()
    # plt.savefig('maeTestingPlots/monthly_hourly_psnr_heatmap.png')
    # plt.show()

    # return heatmap_data


# def plot_lst_heatmaps_test(lst_data, is_target=False, epoch=None, day=None, hour=None, quad_hash=None):
def plot_hourly_lst_change(model, dataloader, device, is_target = None, num_images = 25):
    model.eval()
    
    lst_values = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, temporal_info, spatial_info) in enumerate(dataloader):
            if batch_idx >= num_images: 
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            temporal_info = temporal_info.to(device)
            # spatial_info = spatial_info.to(device)
            outputs = model(inputs, temporal_info)
            # outputs = model(inputs)
            
           
            for i in range(inputs.size(0)): 
                output_sample = outputs[i].cpu().numpy() 

                avg_lst_per_hour = np.mean(output_sample, axis=(1, 2))  # Shape: (24,)
                lst_values.append(avg_lst_per_hour)

    plt.figure(figsize=(10, 6))

    for lst_per_hour in lst_values:
        plt.plot(range(24), lst_per_hour, marker='o')

    plt.title('Hourly Change of Predicted LSTs')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Predicted LST')
    plt.xticks(range(0, 24, 1)) 
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('maeVitResultsTemporal+Spatial/hourly_lstc_chart.png')
    plt.close()

    if is_target:
        plt.savefig(f'maeVitResultsTemporal+Spatial/test_target_{day}_{hour}_{quad_hash}_epoch{epoch+1}.png')
    else:
        plt.savefig(f'maeVitResultsTemporal+Spatial/test_predicted_{day}_{hour}_{quad_hash}_epoch{epoch+1}.png')
    


if __name__ == "__main__":
    model = maeViT()
    model.eval()
    checkpoint_path = "/s/chopin/k/grad/varsh/diurnalModel/maeVitCheckpointTime.pth"
    model, epoch = load_model(checkpoint_path)
    test_loader = prepare_test_data()
  
    print(f"Number of test samples: {len(test_loader.dataset)}")
    print(f"Number of test batches: {len(test_loader)}")

    print("Starting evaluation...")
    # results = evaluate_model(model, test_loader, device)
    # print("Evaluation Results:")
    # print(f"PSNR: {results['PSNR']:.4f}")

    # inputs, targets, spatial_info = next(iter(test_loader))
    # inputs = inputs.to(device)
    # targets = targets.to(device)
    # # temporal_info = temporal_info.to(device)
    # spatial_info = spatial_info.to(device)
    # # outputs = model(inputs, temporal_info, spatial_info)
    # outputs = model(inputs, spatial_info)

    # original_dataset = test_loader.dataset.dataset
    # indices = test_loader.dataset.indices
    # original_index = indices[0]
    # input_path = original_dataset.input_files[original_index]
    # filename = os.path.basename(input_path) 
    # print("Starting evaluation")
    # print(f"outputtensore{outputs.shape}") 
 

    # for batch_idx, (inputs, targets, spatial_info) in enumerate(test_loader):
    #     inputs = inputs.to(device)
    #     targets = targets.to(device)
    #     outputs = model(inputs)

    #     original_dataset = test_loader.dataset.dataset
    #     indices = test_loader.dataset.indices

    #     for i in range(inputs.size(0)):
    #         original_index = indices[batch_idx * test_loader.batch_size + i]
    #         input_path = original_dataset.input_files[original_index]
    #         filename = os.path.basename(input_path)


    # parts = filename.split('_')
    # if len(parts) == 3:
    #     day, hour, quad_hash = parts
    #     quad_hash = quad_hash.split('.')[0]
    #     output_sample = outputs[0].detach().cpu().numpy()
    #     target_sample = targets[0].detach().cpu().numpy()
    #     # plot_lst_heatmaps_test(output_sample, is_target=False, epoch=epoch, day=day, hour=hour, quad_hash=quad_hash)
    #     # plot_lst_heatmaps_test(target_sample, is_target=True, epoch=epoch, day=day, hour=hour, quad_hash=quad_hash)
    # else:
    #     print(f"Unexpected filename format: {filename}")

    # avg_psnr_by_hour = hour_psnr(model, test_loader, device)

    results = evaluate_model(model, test_loader, device)
    print("Evaluation Results:")
    # print(f"PSNR: {results['PSNR']:.4f}")
    # print(f"SSIM: {results['SSIM']:.4f}")
    # print(f"LPIPS: {results['LPIPS']:.4f}")
    # print(f"CC: {results['CC']:.4f}")
    # print(f"RMSE: {results['RMSE']:.4f}")
    print(f"MAE: {results['MAE']:.4f}")
    # print(f"ERGAS: {results['ERGAS']:.4f}")
    # print(f"FID: {results['FID']:.4f}")

   

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(24), avg_psnr_by_hour, marker='o')
    # plt.title('Average PSNR by Hour')
    # plt.xlabel('Hour of the day')
    # plt.ylabel('Avg PSNR per hour')
    # plt.grid(True)
    # plt.xticks(range(24))
    # plt.savefig('maeVitResultsTemporal+Spatial/avg_psnr_by_hour.png')
    # plt.show()
    # plt.close()

    # print("Average PSNR by hour:", avg_psnr_by_hour)

    # avg_psnr_by_month = month_psnr(model, test_loader, device)

    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, 13), avg_psnr_by_month, marker='o')
    # plt.title('Average PSNR by Month')
    # plt.xlabel('Month')
    # plt.ylabel('Avg PSNR')
    # plt.xticks(range(1, 13))
    # plt.grid(True)
    # plt.savefig('maeVitResultsTemporal+Spatial/avg_psnr_by_month.png')
    # plt.show()
    # plt.close()
    
    # # print("Average PSNR by month:", avg_psnr_by_month)
    # # avg_psnr_by_month_hour = monthly_hourly_psnr(model, test_loader, device)
    # avg_psnr_output = monthly_hourly_psnr(model, test_loader, device)
    # lst_values_by_month = plot_hourly_lst_change(model, test_loader, device)


    # test_model(model, test_loader)