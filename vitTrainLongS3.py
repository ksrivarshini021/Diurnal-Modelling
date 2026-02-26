import math
import os
import natsort
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, Dataset
from torch.utils.checkpoint import checkpoint
from torchvision import transforms
import rasterio
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import timm

# from preTrainedHybrid3 import maeViT
from preTrainedSpatial3 import maeViT
# from preTrainedHybrid import CNN_ViT_Base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = list(range(torch.cuda.device_count()))
print(f"Using device: {device}", flush = True)
print(device_ids)

# device0 = torch.device("cuda:0")
# device1 = torch.device("cuda:1")
# print()

def checkpointed_forward(model, input_tensor):
    return checkpoint(model, input_tensor)


class SatelliteImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, valid_days=None):
        """Dataset for paired satellite images."""
        self.input_files, self.target_files = self._get_image_pairs(
            input_dir, target_dir, valid_days
        )

        if not self.input_files:
            raise ValueError("No matching input files found. Check the directories.")

        self.get_global_min_max()

    def _get_image_pairs(self, input_dir, target_dir, valid_days):
        """Retrieve input-target pairs where each input has 24 corresponding target images."""
        input_files = self._get_files(input_dir, valid_days)
        target_files = self._get_files(target_dir, valid_days)

        input_paths = []
        target_paths = {}

        for file_key, input_path in input_files.items():
            parts = file_key.split("_")

            if len(parts) != 3:
                raise ValueError(f"Unexpected file key format: {file_key}")

            day, hour, quad_hash_with_ext = parts
            quad_hash = quad_hash_with_ext.split(".")[0]
            input_paths.append(input_path)

            day_target_files = []
            for target_hour in range(24):
                target_key = f"{day}_{str(target_hour).zfill(2)}_{quad_hash}"
                day_target_files.append(target_files.get(target_key, None))
                # print("day target_key",target_key)
            target_paths[input_path] = day_target_files

        # print("Available target keys:", list(target_files.keys()))

        print(
            f"Found {len(input_paths)} input images with corresponding target sequences."
        )
        # print(target_paths.values())
        return input_paths, target_paths

    def get_global_min_max(self):
        """Compute global min/max values for normalization."""
        self.global_min_lst, self.global_max_lst = float("inf"), float("-inf")
        self.global_min_kopp, self.global_max_kopp = float("inf"), float("-inf")
        self.global_min_elev, self.global_max_elev = float("inf"), float("-inf")

        # self.global_min_target, self.global_max_target = float("inf"), float("-inf")
        to_be_deleted = []
        for input_image_path in self.input_files:
            input_image = self._load_tif_image(input_image_path)
            if input_image is None:
                to_be_deleted.append(input_image_path)
                continue

            input_image[1, input_image[1] == 65535] = float("nan")

            self.global_min_elev = min(
                self.global_min_elev, torch.min(input_image[0]).item()
            )
            self.global_max_elev = max(
                self.global_max_elev, torch.max(input_image[0]).item()
            )
            self.global_min_lst = min(
                self.global_min_lst,
                torch.nan_to_num(input_image[1], nan=float("inf")).min().item(),
            )
            self.global_max_lst = max(
                self.global_max_lst,
                torch.nan_to_num(input_image[1], nan=float("-inf")).max().item(),
            )
            self.global_min_kopp = min(
                self.global_min_kopp, torch.min(input_image[2]).item()
            )
            self.global_max_kopp = max(
                self.global_max_kopp, torch.max(input_image[2]).item()
            )

        self.input_files = list(set(self.input_files) - set(to_be_deleted))

    def _get_files(self, directory, valid_days=None):
        """Retrieve .tif files from a nested directory structure."""
        files = {}

        # Ensure directory exists
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return files

        for day_folder in natsort.natsorted(os.listdir(directory)):
            full_day_path = os.path.join(directory, day_folder)
            
            if not os.path.isdir(full_day_path):
                print(f"Skipping non-directory: {full_day_path}")
                continue

            if valid_days and day_folder not in valid_days:
                # print(f"Skipping day {day_folder} (not in valid_days: {valid_days})")
                continue  # Skip days that are not in valid_days

          

            for hour_folder in natsort.natsorted(os.listdir(full_day_path)):
                full_hour_path = os.path.join(full_day_path, hour_folder)

                if not os.path.isdir(full_hour_path):
                    print(f"Skipping non-directory: {full_hour_path}")
                    continue

                for quad_folder in natsort.natsorted(os.listdir(full_hour_path)):
                    full_quad_path = os.path.join(full_hour_path, quad_folder)

                    if not os.path.isdir(full_quad_path):
                        print(f"Skipping non-directory: {full_quad_path}")
                        continue

                    tif_files = [
                        file
                        for file in os.listdir(full_quad_path)
                        if file.lower().endswith(".tif")
                    ]

                    for tif_file in tif_files:
                        file_key = f"{day_folder}_{hour_folder}_{quad_folder}"
                        file_path = os.path.join(full_quad_path, tif_file)

                        # print(f"      Found .tif file: {file_path} with key {file_key}")

                        files[file_key] = file_path

        return files
    
    def __len__(self):
        # print(f"Total input images: {len(self.input_files)}")
        total_target_images = sum(
            len(target_paths) for target_paths in self.target_files.values()
        )
        return len(self.input_files)

    def _load_tif_image(self, image_path, is_target=False):
        """Load .tif image with rasterio."""
        with rasterio.open(image_path) as dataset:
            num_bands = dataset.count
            if is_target or num_bands == 1:
                target_tens = torch.tensor(
                    dataset.read(1).astype(np.float32), dtype=torch.float32
                )
                return target_tens
            else:
                bands = [dataset.read(i).astype(np.float32) for i in range(1, 4)]
                input_tens = torch.tensor(np.array(bands))
                if torch.min(input_tens[1]) == 65535:
                    return None
                else:
                    return input_tens
                
    def calculate_longitude(image_path):
        with rasterio.open(image_path) as dataset:
            width, height = dataset.width, dataset.height
            centerRow, centerCol = height // 2, width // 2
            lon, lat = dataset.xy(centerRow, centerCol)
        return lon

    def __getitem__(self, index):
        """Retrieve the input image and its 24 corresponding target images."""
        input_image_path = self.input_files[index]
        target_image_paths = self.target_files[input_image_path]

        # print(f"Loading input image: {input_image_path}")
        # print(f"Number of target images for {input_image_path}: {len(target_image_paths)}")

        input_image = self._load_tif_image(input_image_path)
        input_image[1, input_image[1] == 65535] = float("inf")

        input_image[0] = (input_image[0] - self.global_min_elev) / (
            self.global_max_elev - self.global_min_elev
        )
        input_image[1] = (input_image[1] - self.global_min_lst) / (
            self.global_max_lst - self.global_min_lst
        )
        input_image[2] = (input_image[2] - self.global_min_kopp) / (
            self.global_max_kopp - self.global_min_kopp
        )
        input_image[torch.isinf(input_image)] = -1

        target_images = []

        for target_path in target_image_paths:
            if target_path is None:
                target_images.append(torch.full((32, 32), -1, dtype=torch.float32))
            else:
                target_image = self._load_tif_image(target_path, is_target=True)
                target_image[target_image == 65535] = float("inf")
                target_image = (target_image - self.global_min_lst) / (
                    self.global_max_lst - self.global_min_lst
                )
                target_image[torch.isinf(target_image)] = -1

                if target_image.shape != (32, 32):  
                    target_image = F.interpolate(target_image.unsqueeze(0).unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)

                target_images.append(target_image)

        target_tensor = torch.stack(target_images, dim = 0)
        #---timesptams code

        filename = os.path.basename(input_image_path)
        day, hour, _ = filename.split("_")
        dayValue = int(day)
        hourValue = int(hour)

        """day_cycle = 2 * math.pi * (dayValue / 365.0)
        hour_cycle = 2 * math.pi * (hourValue / 24.0)

        day_sin = math.sin(day_cycle)
        hour_sin = math.sin(hour_cycle)
        day_cos = math.cos(day_cycle)
        hour_cos = math.cos(hour_cycle)"""

        with rasterio.open(input_image_path) as dataset:
            width, height = dataset.width, dataset.height
            centerRow, centerCol = height // 2, width // 2
            lon, lat = dataset.xy(centerRow, centerCol)

            # print degugging stats
            # print(f"Image: {input_image_path}")
            # print(f"Center pixel (row{centerRow}, col = {centerCol})")
            # print(f" longitude: {lon:.4f}")
            # print(f"width: {width}, heighrt{height}")
            # print(f"CRS{dataset.crs}")
            # print(f"Transofrm:{dataset.transform}")

        solar_time = (hourValue + (lon / 15.0)) % 24

        day_angle = 2 * math.pi * (dayValue / 365.0)
        solar_angle = 2 * math.pi * (solar_time / 24.0)

        # temporal_info = torch.tensor([math.sin(day_angle), math.cos(day_angle), math.sin(solar_angle), math.cos(solar_angle)], dtype = torch.float32). view(1, -1)

        # print degugging stats
        # print(f"image: {input_image_path}")
        # print(f"Hour, {hourValue}")
        # print(f"longitude{lon:.4f}")
        # print(f"Solar time{solar_time:.4f}")

        # temporal_info = torch.tensor([day_sin, day_cos, hour_sin, hour_cos], dtype=torch.float32)
        # print(f"1, {temporal_info.shape}")
        # temporal_info = torch.tensor([day_sin, day_cos, hour_sin, hour_cos], dtype=torch.float32).unsqueeze(0)
        # temporal_info = temporal_info.view(1, -1)
        # print(f"2, {temporal_info.shape}")

        # ----Spatial info

        lat_angle = 2 * math.pi * (lat/180.0) # lat- [-90, 90]
        lon_angle = 2 * math.pi * (lon/360.0) # lon- [-18-, 180]

        spatial_info = torch.tensor([math.sin(lat_angle), math.cos(lat_angle), math.sin(lon_angle), math.cos(lon_angle)], dtype = torch.float32). view(1, -1)

        # print degugging stats
        # print(f"image: {input_image_path}")
        # print(f"lat, lat_angle{lat:.4f},{lat_angle}")
        # print(f"longitude, lon_angle{lon:.4f},{lon_angle}")

        return input_image, target_tensor, spatial_info     

        # for idx, img in enumerate(target_images):
        #     print(f"Target image {idx} shape: {img.shape}")

        target_tensor = torch.stack(target_images, dim=0)
        return input_image, target_tensor


# global min max max for all the three bands
# global min for entire data
# normalized_data = (data - global_mean) / global_std

# check nanmean- mean ; nanstd- stand deviation
# for 0 to 1 range use global min nd max formula- done
# pinrt statwmnwt to see normalized vaule; normalized value for entire files- min and max. -done
# select value outside normalized set; close to therange
# correct calucation of loss function


# def masked_mse_loss(pred, target, mask_value=-1):
#     """
#     predicted target output: [batch_size, 24, 32, 32]
#     target-ground truth: [batch_size, 24, 32, 32]
#     """

#     mask = (target != mask_value).float().to(device)
#     if mask.sum() == 0:
#         print("Warning: All values are masked! Returning zero loss.")
#         # return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
#         return torch.tensor(0.0, dtype=pred.dtype, device=device)

#     # pred = pred[mask]
#     # target = target[mask]
#     pred = pred * mask
#     target = target * mask

#     loss = torch.mean((pred - target) ** 2)

#     # print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum().item()} (valid pixels)")
#     return loss

def masked_mae_loss(pred, target, mask_value=-1, lamda = 0.0001):
    mask = (target != mask_value).float().to(device)
    if mask.sum() == 0:
        print("Warning: All values are masked! Returning zero loss.")
        return torch.tensor(0.0, dtype=pred.dtype, device=device)

    pred = pred * mask
    target = target * mask

    loss = torch.sum(torch.abs(pred - target)) / mask.sum()
    #for smoothing
    total_var_height = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]).mean()
    total_var_width = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]).mean()
    var_loss = total_var_height + total_var_width

    total_var_loss = loss + lamda * var_loss
    return total_var_loss



def masked_huber_loss(pred, target, mask_value=-1, delta=1.0):
    device = pred.device 
    mask = (target != mask_value).float().to(device) 

    if mask.sum() == 0:
        print("Warning: All values are masked! Returning zero loss.")
        return torch.tensor(0.0, dtype=pred.dtype, device=device)

    abs_error = torch.abs(pred - target)
    
    quadratic = 0.5 * (abs_error ** 2)
    linear = delta * (abs_error - 0.5 * delta)
    
    huber_loss = torch.where(abs_error <= delta, quadratic, linear)
    
    huber_loss = huber_loss * mask
    loss = huber_loss.sum() / mask.sum()  
    return loss


# def calculate_mape(pred, target, mask_value=-1):
#     mask = target != mask_value
#     valid_pred = pred[mask]
#     valid_target = target[mask]

#     if valid_target.numel() == 0:
#         return 0.0

#     safe_target = torch.where(valid_target == 0, torch.tensor(1e-6, device=target.device), valid_target)

#     mape = torch.mean(torch.abs((valid_pred - valid_target) / safe_target)) * 100
#     return mape.item()


def plot_training_results(train_losses, test_losses, train_psnrs, test_psnrs):
    if not train_losses or not test_losses or not train_psnrs or not test_psnrs:
        print("Error: One or more input lists are empty, skipping plot.")
        return

    train_losses = [loss.cpu().detach().item() if torch.is_tensor(loss) else loss for loss in train_losses]
    test_losses = [loss.cpu().detach().item() if torch.is_tensor(loss) else loss for loss in test_losses]
    train_psnrs = [psnr.cpu().detach().item() if torch.is_tensor(psnr) else psnr for psnr in train_psnrs]
    test_psnrs = [psnr.cpu().detach().item() if torch.is_tensor(psnr) else psnr for psnr in test_psnrs]

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Train and Test Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Testing Loss")
    plt.legend()
    plt.grid()

    # Plot Train and Test PSNR (Accuracy)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, label="Train PSNR", marker="o")
    plt.plot(epochs, test_psnrs, label="Test PSNR", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR (dB)")
    plt.title("Training & Testing PSNR")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    save_path = 'maeVitResultsSpatial/Train-TestGraph2.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    plt.show(block=True)  
    plt.close()

    print('Train and test loss and accuracy plotted successfully!')


    # print('Train and test loss and accuracy plotted successfully!')


    # plt.tight_layout()  
    # if save_path:
    #     plt.savefig(f'./Train-TestGraph.png') 
    #     print(f"Plot saved to {save_path}")
    # else:
    #     plt.show()




def plot_lst_heatmaps(lst_data, is_target=False, epoch=None):
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    fig.suptitle(f'Land Surface Temperature (24 Hours)', fontsize=16)

    lst_data = torch.tensor(lst_data)
    lst_data = lst_data.cpu().detach().numpy()
    if lst_data.size == 0:
        print(f"Warning: lst_data is empty for epoch {epoch}, batch {batch_idx}")
        return  
    lst_data = lst_data[0] 
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
        plt.savefig(f'maeVitResultsSpatial/target_epoch{epoch+1}.png')
    else:
        plt.savefig(f'maeVitResultsSpatial/predicted_epoch{epoch+1}.png')
    plt.close()


def calculate_psnr(pred, target, mask_value=-1, max_pixel_value=1.0):
    # mask = (target != mask_value)
    # valid_pred = pred[mask]
    # valid_target = target[mask]

    mask = (target != mask_value).float().to(device)
    valid_pred = pred * mask
    valid_target = target * mask

    # if valid_target.numel() == 0:
    #     print("No data in target")
    #     return 0.0

    mse = torch.mean(torch.abs(valid_pred - valid_target) ** 2)
    psnr = 20 * torch.log10(
        torch.tensor(max_pixel_value, device=target.device)
    ) - 10 * torch.log10(mse)    
    return psnr.item()

def test_model(model, test_dataloader, epoch=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    total_psnr = 0.0

    # printed_plot = False  
    with torch.no_grad():
        for batch_idx, (inputs, targets, sptaial_info) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            sptaial_info =  sptaial_info.to(device)
            outputs = model(inputs, sptaial_info)

            loss = masked_mae_loss(outputs, targets)
            psnr = calculate_psnr(outputs, targets)

            total_loss += loss.item()
            total_psnr += psnr
            num_batches += 1

            # if not printed_plot:
            #     plot_lst_heatmaps(outputs, is_target=False, epoch=epoch)
            #     plot_lst_heatmaps(targets, is_target=True, epoch=epoch)
            #     printed_plot = True 

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0

    print(f"Test Loss: {avg_loss:.4f}, Test PSNR: {avg_psnr:.2f} dB")
    return avg_loss, avg_psnr

def save_checkpoint(model, optimizer, epoch, train_losses, test_losses, filename="maeVitCheckpointSpatial.pth"):

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch+1} to {filename}")


def load_checkpoint(model, optimizer, filename="maeVitCheckpointSpatial.pth"):
   
    if os.path.exists(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        train_losses = checkpoint.get('train_losses', [])
        test_losses = checkpoint.get('test_losses', [])
        print(f"Checkpoint loaded: Resuming from epoch {start_epoch+1}")
        return start_epoch, train_losses, test_losses
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, [], []

def train_model(model, train_dataloader, test_dataloader, num_epochs, learning_rate, mask_threshold = 0.9, checkpoint_path="maeVitCheckpointSpatail.pth"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch, train_losses, test_losses = load_checkpoint(model, optimizer, checkpoint_path)

    train_psnrs = []
    test_psnrs = []

    print(f"Started training from epoch: {start_epoch + 1}")
    print(f"Starting training with {len(train_dataloader)} batches...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets, spatial_info) in enumerate(train_dataloader):
            # print(f"bathc:{batch_idx}")
            # if torch.sum(targets != -1) == 0:
            #     print(f"Skipping batch {batch_idx + 1}, all target values are missing.")
            #     continue

            num_total_values = targets.numel()
            num_masked_values = torch.sum(targets == -1).item()
            masked_percentage = num_masked_values / num_total_values

            if masked_percentage > mask_threshold:
                print(f"Skipping batch {batch_idx + 1} due to high masked data (masked percentage: {masked_percentage*100:.2f}%)")
                continue

            inputs, targets = inputs.to(device), targets.to(device),
            spatial_info =  spatial_info.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, spatial_info)

            loss = masked_mae_loss(outputs, targets)
            psnr = calculate_psnr(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_psnr += psnr
            num_batches += 1


            # if (epoch +1) % 10 == 0:
            #     plot_lst_heatmaps(outputs, is_target=False, epoch=epoch)
            #     plot_lst_heatmaps(targets, is_target=True, epoch=epoch)

        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_train_psnr = running_psnr / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train PSNR: {avg_train_psnr:.2f} dB")

        print(f"Testing after epoch {epoch + 1}...")
        test_loss, test_psnr = test_model(model, test_dataloader,epoch)
        test_losses.append(test_loss)
        test_psnrs.append(test_psnr)

        # if (epoch +1) % 10 == 0:
        # plot_lst_heatmaps(outputs, is_target=False, epoch=epoch)
        # plot_lst_heatmaps(targets, is_target=True, epoch=epoch)
        # plot_training_results(train_losses, test_losses, train_psnrs, test_psnrs)

        # if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch, train_losses, test_losses, filename = checkpoint_path)


        # Test on testing data only
        # test_model()
        # if epoch % 10 == 0:
        #     save_model_prediction(sample_image_from_test_data, target_24_hrs)
        #       put in title geohash, day, MSE
        #     save model weights

        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f} dB", flush = True)
        

    print("Training completed!")
    plot_training_results(train_losses, test_losses, train_psnrs, test_psnrs)
    # plot_lst_heatmaps(last_outputs, is_target=False, epoch=epoch)
    # plot_lst_heatmaps(last_targets, is_target=True, epoch=epoch)

def perform_inference(saved_model_path, test_loader, save_plot=True):
    # from opath lodad svaed trained model
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    printed_plot = False

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu())

            # if save_plot and not printed_plot:
            #     try:
            #         # plot_lst_heatmaps(outputs, is_target=False, epoch="none")
            #         # plot_lst_heatmaps(targets, is_target=True, epoch="none")

            #     except Exception as e:
            #         print(f"Plotting failed: {e}")
            #     printed_plot=True
    
    all_predictions = torch.cat(predictions, dim=0)
    print("inference completed")
    return all_predictions

if __name__ == "__main__":
    input_dir = (
        "/s/chopin/e/proj/hyperspec/diurnalModel/combinedQuadHash2/"
    )
    target_dir = (
        "/s/chopin/e/proj/hyperspec/diurnalModel/LSTCTargetTest/"
    )

    # valid_days = ['001','002']

    valid_days = os.listdir(input_dir)
    valid_days = valid_days[:1]

    dataset = SatelliteImageDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        valid_days=valid_days,
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 64
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        # , num_workers=4
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 drop_last = True)
    num_workers = 4

    # config = {
    #     "image_size": 32,
    #     "patch_size": 16,
    #     "num_channels": 3,
    #     "hidden_size": 768,
    #     "num_outputs": 24,
    #     "num_attention_heads": 8,
    #     "num_hidden_layers": 12,
    #     "intermediate_size": 3072,
    #     "hidden_dropout_prob": 0.1,
    #     "layer_norm_eps": 1e-6,
    #     "batch_size": batch_size,
    # }

    # model = ViTArch(config)
    # model = ViTRegressor(config)
    # model = HybridCNNViT(config)

    # model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False

    # model.set_input_size(img_size=(32, 32), patch_size=(8, 8))

    # model.head = nn.Sequential(
    #     nn.Linear(192, 24 * 32 * 32), 
    #     nn.Unflatten(1, (24, 32, 32))
    # )

    model = maeViT()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
        print("using gpus")

    train_model(
    model, train_dataloader, test_dataloader, num_epochs=1, learning_rate=0.001
)

# torch.save(model.state_dict(), "vit_regressor_trained.pth")
# torch.save(model.state_dict(), "vitArch_trained.pth")
# saved_model_path = "vitArch_trained.pth"
# print(f"Trained model saved as {saved_model_path}")
# perform_inference(saved_model_path, test_dataloader, save_plot=True)
# to perform inference
    #  load trained model
   

    # "rasterio stuff"
    # import rasterio
    # import numpy as np

    # def load_image_with_rasterio(image_path):
    #     with rasterio.open(image_path) as src:
    #         bands = [src.read(i) for i in range(1, 4)]  # Read the first 3 bands
    #         image = np.stack(bands, axis=-1)
    #     return image

    # image_path = "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/comdinedQuadHash/012/19/030232202/012_19_030232202.tif"
    # image = load_image_with_rasterio(image_path)

    # if image is not None:
    #     print(f"Image shape: {image.shape}")
