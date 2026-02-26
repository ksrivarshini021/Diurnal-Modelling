import os
import natsort
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, Dataset
from torchvision import transforms
import rasterio
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import VitArchRevised
import matplotlib.pyplot as plt
import torch.nn as nn

# from VitArchRevised import ViTArch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SatelliteImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, valid_days=None):
        """Dataset for paired satellite images."""
        self.input_files, self.target_files = self._get_image_pairs(
            input_dir, target_dir, valid_days
        )

        if not self.input_files:
            raise ValueError("No matching input files found. Check the directories.")

        self.get_global_min_max()

        # SPLIT DATA IN TRAIN TEST
        # self.input_files

    def _get_image_pairs(self, input_dir, target_dir, valid_days):
        """Retrieve input-target pairs where each input has 24 corresponding target images."""
        input_files = self._get_files(input_dir, valid_days)
        target_files = self._get_files(target_dir, valid_days)

        input_paths = []
        target_paths = {}

        for file_key, input_path in input_files.items():
            # print(f"Processing file key: {file_key}")
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
            # print("day_target_files:", day_target_files)
            target_paths[input_path] = day_target_files

        # print("Available target keys:", list(target_files.keys())[:10])

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

        # print("Items to be deleted: ", len(to_be_deleted))
        self.input_files = list(set(self.input_files) - set(to_be_deleted))

        # for paths in self.target_files.keys():
        #     if paths not in self.input_files:
        #         continue
        #     target_image_paths = self.target_files.get(paths)
        #     for target_image_path in target_image_paths:
        #         if target_image_path is not None:
        #             target_image = self._load_tif_image(
        #                 target_image_path, is_target=True
        #             )
        #             print("target_image.", target_image.shape)

        #             target_image[target_image == 65535] = float("nan")
        #             self.global_min_target = min(
        #                 self.global_min_lst,
        #                 torch.nan_to_num(target_image, nan=float("inf")).min().item(),
        #             )
        #             self.global_max_target = max(
        #                 self.global_max_target,
        #                 torch.nan_to_num(target_image, nan=float("-inf")).max().item(),
        #             )

    def _get_files(self, directory, valid_days=None):
        """Retrieve .tif files from a nested directory structure."""
        files = {}
        for day_folder in natsort.natsorted(os.listdir(directory)):
            if valid_days and day_folder not in valid_days:
                continue
            for hour_folder in natsort.natsorted(
                os.listdir(os.path.join(directory, day_folder))
            ):
                for quad_folder in natsort.natsorted(
                    os.listdir(os.path.join(directory, day_folder, hour_folder))
                ):
                    quad_path = os.path.join(
                        directory, day_folder, hour_folder, quad_folder
                    )
                    tif_files = [
                        file
                        for file in os.listdir(quad_path)
                        if file.lower().endswith(".tif")
                    ]
                    if tif_files:
                        file_key = f"{day_folder}_{hour_folder}_{quad_folder}"
                        files[file_key] = os.path.join(quad_path, tif_files[0])

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
                target_images.append(target_image)

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


def masked_mse_loss(pred, target, mask_value=-1):
    """
    predicted target output: [batch_size, 24, 32, 32]
    target-ground truth: [batch_size, 24, 32, 32]
    """

    mask = (target != mask_value).float()
    if mask.sum() == 0:
        print("Warning: All values are masked! Returning zero loss.")
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)

    # pred = pred[mask]
    # target = target[mask]
    pred = pred * mask
    target = target * mask

    loss = torch.mean((pred - target) ** 2)

    # print(f"Mask shape: {mask.shape}, Mask sum: {mask.sum().item()} (valid pixels)")
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
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    #same plot
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, test_losses, label="Test Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Testing Loss")
    plt.legend()
    plt.grid()


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_psnrs, label="Train PSNR", marker="o")
    plt.plot(epochs, test_psnrs, label="Test PSNR", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR (dB)")
    plt.title("Training & Testing PSNR")
    plt.legend()
    plt.grid()

    plt.show()


def plot_lst_heatmaps(lst_data, is_target=False):
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    fig.suptitle('Land Surface Temperature (24 Hours)', fontsize=16)
    lst_data = lst_data.detach().numpy()
    lst_data = lst_data[0]
    masked_data = np.ma.masked_where(lst_data < 0, lst_data)

    for hour in range(24):
        ax = axes[hour // 6, hour % 6]
        cax = ax.imshow(masked_data[hour], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Hour {hour:02d}')
        ax.axis('off')

    fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.02, pad=0.05, label='Temperature')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if is_target:
        plt.savefig('./sample_image_target.png')
    else:
        plt.savefig('./sample_image_predicted.png')
    print("Plot saved for test file")
    plt.close()


def calculate_psnr(pred, target, mask_value=-1, max_pixel_value=1.0):
    # mask = (target != mask_value)
    # valid_pred = pred[mask]
    # valid_target = target[mask]

    mask = (target != mask_value).float()
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

def test_model(model, test_dataloader):
    model.eval()
    total_loss = 0.0
    # total_mape = 0.0
    num_batches = 0
    total_psnr = 0.0

    printed_plot = False
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # where target are -1, and set those pixcel in output to -1 as well
            if printed_plot is False:
                plot_lst_heatmaps(outputs, is_target=False)
                plot_lst_heatmaps(targets, is_target=True)
                printed_plot = True

            loss = masked_mse_loss(outputs, targets)
            # mape = calculate_mape(outputs, targets)
            psnr = calculate_psnr(outputs, targets)

            total_loss += loss.item()
            # total_mape += mape
            total_psnr += psnr
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    # avg_mape = total_mape / num_batches if num_batches > 0 else 0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0

    # print(f"Test Loss: {avg_loss:.4f}, Test Accuracy : {avg_accuracy:.2f}%")
    # return avg_loss, avg_accuracy

    print(f"Test Loss: {avg_loss:.4f}, Test PSNR: {avg_psnr:.2f} dB")
    return avg_loss, avg_psnr


def train_model(model, train_dataloader, test_dataloader, num_epochs, learning_rate):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_psnrs = []
    test_psnrs = []

    print(f"Starting training with {len(train_dataloader)} batches...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            if torch.sum(targets != -1) == 0:
                print(f"Skipping batch {batch_idx + 1}, all target values are missing.")
                continue

            inputs, targets = (inputs.to(device), targets.to(device))
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = masked_mse_loss(outputs, targets)
            psnr = calculate_psnr(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_psnr += psnr
            num_batches += 1

            plot_lst_heatmaps(outputs, is_target=False)

        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        avg_train_psnr = running_psnr / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        train_psnrs.append(avg_train_psnr)
        

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train PSNR: {avg_train_psnr:.2f} dB")

        print(f"Testing after epoch {epoch + 1}...")
        test_loss, test_psnr = test_model(model, test_dataloader)
        test_losses.append(test_loss)
        test_psnrs.append(test_psnr)

        # Test on testing data only
        # test_model()
        # if epoch % 10 == 0:
        #     save_model_prediction(sample_image_from_test_data, target_24_hrs)
        #       put in title geohash, day, MSE
        #     save model weights

        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f} dB")

    print("Training completed!")
    plot_training_results(train_losses, test_losses, train_psnrs, test_psnrs)
    # plot_lst_heatmaps(outputs, is_target=False)
    # plot_lst_heatmaps(targets, is_target=True)


if __name__ == "__main__":
    input_dir = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/combinedQuadHash/"
    )
    target_dir = (
        "/s/lattice-151/a/all/all/all/sustain/varsh/Python/GOES/quadHashTest32x32/"
    )

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

    batch_size = 16
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        # , num_workers=4
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                 drop_last = True)
    num_workers = 4
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,
    # )

    config = {
        "image_size": 32,
        "patch_size": 16,
        "num_channels": 3,
        "hidden_size": 768,
        "num_outputs": 24,
        "num_attention_heads": 8,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "hidden_dropout_prob": 0.1,
        "layer_norm_eps": 1e-6,
        "batch_size": batch_size,
    }

    model = VitArchRevised.ViTArch(config)
    train_model(
        model, train_dataloader, test_dataloader, num_epochs=10, learning_rate=0.0001
    )

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
