import os
import random
import math
import torch
import rasterio
import numpy as np
from torch.utils.data import random_split, DataLoader
from collections import defaultdict
import mercantile
from preTrainedTemporalSpatial3 import maeViT
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Land cover mapping (NLCD)
# -----------------------------
landcover_map = {
    11: "Open Water",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/Clay)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
}

# -----------------------------
# 1. Load model from checkpoint
# -----------------------------
def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = maeViT()
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    return model

# -----------------------------
# 2. Input tensor preparation
# -----------------------------
def input_tensor(tif_path):
    """
    Returns:
      - tensor: torch tensor shape (1, bands, H, W) on device
      - lst_min, lst_max: computed from band 2 (index 1) excluding sentinel NaNs
      - hw: (height, width) of raster
    """
    with rasterio.open(tif_path) as src:
        array = src.read().astype(np.float32)  # (bands, H, W)
        # treat sentinel values as NaN for LST band (index 1)
        array[1][array[1] == 65535] = np.nan

        # compute min/max for band 1 excluding NaN (useful for scaling or troubleshooting)
        lst_band = array[1]
        finite = np.isfinite(lst_band)
        if np.any(finite):
            lst_min = float(np.nanmin(lst_band))
            lst_max = float(np.nanmax(lst_band))
        else:
            lst_min, lst_max = 0.0, 1.0

    tensor = torch.tensor(array, dtype=torch.float32).unsqueeze(0)  # (1, bands, H, W)
    return tensor.to(device), lst_min, lst_max, (array.shape[1], array.shape[2])

# -----------------------------
# 3. Temporal + spatial info
# -----------------------------
def get_temporal_info(day: str, hour: str, tif_path: str):
    day_value = int(day)   # expected day-of-year or day index already
    hour_value = int(hour)
    with rasterio.open(tif_path) as dataset:
        width, height = dataset.width, dataset.height
        center_row, center_col = height // 2, width // 2
        lon, _ = dataset.xy(center_row, center_col)
    solar_time = (hour_value + (lon / 15.0)) % 24
    day_angle = 2 * math.pi * (day_value / 365.0)
    solar_angle = 2 * math.pi * (solar_time / 24.0)
    return torch.tensor([
        math.sin(day_angle),
        math.cos(day_angle),
        math.sin(solar_angle),
        math.cos(solar_angle)
    ], dtype=torch.float32).view(1, -1).to(device)

def get_spatial_info(quadkey):
    tile = mercantile.quadkey_to_tile(quadkey)
    bbox = mercantile.bounds(tile)
    center_lon = (bbox.west + bbox.east) / 2
    center_lat = (bbox.north + bbox.south) / 2
    lat_norm = center_lat / 90.0
    lon_norm = center_lon / 180.0
    spatial_vec = [lat_norm, lon_norm,
                   np.sin(np.pi * lat_norm), np.cos(np.pi * lon_norm)]
    return torch.tensor(spatial_vec, dtype=torch.float32).unsqueeze(0).to(device)

# -----------------------------
# 4. Predict tile
# -----------------------------
def predict_tile(model, inputs, temporal_info, spatial_info):
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, temporal_info, spatial_info)
        return outputs.squeeze(0).cpu().numpy()  # expected (24, H, W)

# -----------------------------
# 5. Save as GeoTIFF
# -----------------------------
def save_tif(output_path, array, reference_path):
    # array: 2D numpy (H, W)
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        # write as single-band float32
        profile.update(dtype=rasterio.float32, count=1, compress="lzw")
        # ensure transform, crs, width/height kept or updated automatically by rasterio
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)

# -----------------------------
# 6. Load landcover tile
# -----------------------------
def get_landcover_for_tile(quadkey, landcover_root, target_shape=None):
    """
    target_shape: (height, width) or None. If provided, landcover is resized to that shape.
    """
    lc_path = os.path.join(landcover_root, quadkey, "landcover.tif")
    if not os.path.exists(lc_path):
        return None
    with rasterio.open(lc_path) as src:
        lc_arr = src.read(1)
    if target_shape is not None:
        # cv2.resize expects (width, height)
        h, w = target_shape
        lc_small = cv2.resize(lc_arr, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        lc_small = cv2.resize(lc_arr, (32, 32), interpolation=cv2.INTER_NEAREST)
    return lc_small

# -----------------------------
# 7. Robust PSNR calculation
# -----------------------------
def calculate_psnr(pred, target, mask=None, max_pixel_value=None, zero_mse_tol=1e-12):
    """
    Robust PSNR:
      - pred, target: numpy arrays (H, W)
      - mask: boolean array True for pixels to include (same shape as target). If None, uses ~np.isnan(target).
      - max_pixel_value: peak amplitude. If None, uses target.ptp() (peak-to-peak). If that is 0, uses abs(max target).
      - Returns float('inf') if mse <= zero_mse_tol, float('nan') if no valid pixels.
    """
    pred = np.asarray(pred)
    target = np.asarray(target)

    if mask is None:
        mask = np.isfinite(target)  # include only finite target pixels
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(target)

    if mask.shape != target.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match target shape {target.shape}")

    valid_pred = pred[mask]
    valid_target = target[mask]
    if valid_target.size == 0:
        return float('nan')

    mse = np.mean((valid_pred - valid_target) ** 2)

    if mse <= zero_mse_tol:
        return float('inf')

    # determine peak amplitude
    if max_pixel_value is None:
        p2p = valid_target.max() - valid_target.min()
        if p2p > 0:
            max_pixel_value = p2p
        else:
            max_pixel_value = np.max(np.abs(valid_target))
            if max_pixel_value == 0:
                return float('nan')

    # Standard formula: 10 * log10( MAX^2 / MSE )
    psnr = 10.0 * np.log10((max_pixel_value ** 2) / mse)
    return float(psnr)

# -----------------------------
# 8. Predict days and compute PSNR per landcover
# -----------------------------
def predict_days(model, input_dir, target_dir, predicted_root, landcover_root, fraction=0.2):
    all_days = sorted(os.listdir(input_dir))
    test_days = sorted(os.listdir(target_dir))
    valid_days = list(set(all_days) & set(test_days))
    valid_days.sort()
    print(f"Found {len(valid_days)} common test days")

    if len(valid_days) == 0:
        print("No matching days found between input and target directories.")
        return

    num_days = max(1, int(len(valid_days) * fraction))
    if num_days > len(valid_days):
        num_days = len(valid_days)
    chosen_days = random.sample(valid_days, num_days)
    print(f"Selected {num_days} days for prediction")

    psnr_dict = defaultdict(list)

    for day in chosen_days:
        day_path = os.path.join(input_dir, day)
        if not os.path.isdir(day_path):
            continue

        quadhashes = sorted(os.listdir(day_path))
        for qh in quadhashes:
            quad_path = os.path.join(day_path, qh)
            if not os.path.isdir(quad_path):
                continue

            tif_files = [f for f in os.listdir(quad_path) if f.endswith(".tif")]
            if not tif_files:
                continue
            tif_path = os.path.join(quad_path, tif_files[0])

            # prepare input tensor (includes lst_min/lst_max and raster shape)
            inputs, lst_min, lst_max, (H, W) = input_tensor(tif_path)

            temporal_info = get_temporal_info(day, "0", tif_path)
            spatial_info = get_spatial_info(qh)

            predictions = predict_tile(model, inputs, temporal_info, spatial_info)
            # predictions expected shape (24, H, W). If shape mismatches (rare), we'll try to resize below.

            # Save each hour (resizing if needed to match reference raster)
            out_day_dir = os.path.join(predicted_root, day, qh)
            os.makedirs(out_day_dir, exist_ok=True)
            for h in range(24):
                pred_h = predictions[h]
                # if pred shape doesn't match raster H,W, resize with bilinear
                if pred_h.shape != (H, W):
                    pred_h_resized = cv2.resize(pred_h, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    pred_h_resized = pred_h
                out_path = os.path.join(out_day_dir, f"{h:02d}.tif")
                save_tif(out_path, pred_h_resized, tif_path)

            # Compute PSNR per land cover using the first reference ground-truth file in the quad folder
            # open the corresponding ground truth file from target_dir: try to find same structure
            # We'll assume target_dir has matching day/quad structure; else use the local gt file in this quad_path if present
            # Try to locate gt file for this day/quad in target_dir
            gt_file = None
            candidate = os.path.join(target_dir, day, qh)
            if os.path.isdir(candidate):
                gt_files = [f for f in os.listdir(candidate) if f.endswith(".tif")]
                if gt_files:
                    gt_file = os.path.join(candidate, gt_files[0])
            # fallback: try local quad_path
            if gt_file is None:
                local_gt = os.path.join(quad_path, tif_files[0])
                if os.path.exists(local_gt):
                    gt_file = local_gt

            if gt_file is None:
                print(f"WARNING: no ground truth found for day {day} quad {qh}; skipping PSNR.")
                continue

            with rasterio.open(gt_file) as src:
                # read LST band (band index 2 in your previous code). We'll guard if less bands exist.
                if src.count >= 2:
                    gt_arr = src.read(2).astype(np.float32)
                else:
                    gt_arr = src.read(1).astype(np.float32)

            # Resize landcover to GT shape
            lc_tile = get_landcover_for_tile(qh, landcover_root, target_shape=gt_arr.shape)
            if lc_tile is None:
                # no landcover; skip per-landcover PSNR but optionally compute global PSNR
                # Compute global PSNR for hour 0 only (as you previously used predictions[0])
                mask_valid = np.isfinite(gt_arr)
                global_psnr = calculate_psnr(predictions[0], gt_arr, mask=mask_valid, max_pixel_value=(lst_max - lst_min))
                if not np.isnan(global_psnr):
                    psnr_dict[-1].append(global_psnr)  # -1 for global/no-landcover
                continue

            for lc_val in np.unique(lc_tile):
                if lc_val == 0:
                    continue
                # build boolean mask in GT shape
                mask = (lc_tile == lc_val)
                # combine with valid-target mask (exclude NaNs in GT)
                valid_mask = mask & np.isfinite(gt_arr)
                pixel_count = int(np.sum(valid_mask))
                if pixel_count == 0:
                    # nothing to compute for this landcover in this tile
                    continue

                # Choose the prediction hour array to compare — you used predictions[0] previously; keep that behavior
                pred_arr = predictions[0]
                # if pred_arr shape doesn't match gt_arr, resize
                if pred_arr.shape != gt_arr.shape:
                    pred_arr = cv2.resize(pred_arr, (gt_arr.shape[1], gt_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

                # prefer using dynamic range of LST in tile as peak amplitude
                max_val = (lst_max - lst_min) if (lst_max - lst_min) > 0 else None
                psnr_val = calculate_psnr(pred_arr, gt_arr, mask=valid_mask, max_pixel_value=max_val)

                if not np.isnan(psnr_val):
                    psnr_dict[int(lc_val)].append(psnr_val)

                # debug/log suspicious values
                if np.isinf(psnr_val) or (psnr_val is not None and psnr_val > 60):
                    # compute mse for debug
                    valid_pred_pixels = pred_arr[valid_mask]
                    valid_gt_pixels = gt_arr[valid_mask]
                    mse = np.mean((valid_pred_pixels - valid_gt_pixels) ** 2) if valid_gt_pixels.size > 0 else np.nan
                    print(f"DEBUG: day={day} quad={qh} lc={int(lc_val)} pixels={pixel_count} mse={mse:.6e} psnr={psnr_val}")

        print(f"Finished predictions for day {day}")

    # -------------------------
    # Print final PSNR summary per landcover
    # -------------------------
    print("\n=== Final PSNR Summary per Land Cover Class ===")
    for lc_val, psnrs in psnr_dict.items():
        arr = np.array(psnrs, dtype=np.float64)
        # drop inf values (perfect matches) before averaging
        finite_arr = arr[np.isfinite(arr)]
        if finite_arr.size == 0:
            avg_psnr = float("inf") if arr.size > 0 else float("nan")
        else:
            avg_psnr = finite_arr.mean()
        if lc_val == -1:
            label = "Global/NoLandcover"
        else:
            label = landcover_map.get(lc_val, f"LC_{lc_val}")
        if np.isfinite(avg_psnr):
            print(f"{label}: {avg_psnr:.2f} dB (n={finite_arr.size})")
        else:
            print(f"{label}: {avg_psnr} (n_total={len(arr)})")

# -----------------------------
# 9. Run
# -----------------------------
if __name__ == "__main__":
    checkpoint_path = "maeVitCheckpointTimeSolar+sptail.pth"
    input_dir = "/s/chopin/e/proj/hyperspec/diurnalModel/combinedQuadHash2"
    target_dir = "/s/chopin/e/proj/hyperspec/diurnalModel/LSTCTargetTest"
    predicted_root = "/s/chopin/e/proj/hyperspec/diurnalModel/Predicted"
    landcover_root = "/s/chopin/e/proj/hyperspec/diurnalModel/LandCoverQuadHash"

    model = load_model(checkpoint_path)
    predict_days(model, input_dir, target_dir, predicted_root, landcover_root, fraction=0.5)

    print("Prediction complete.")
