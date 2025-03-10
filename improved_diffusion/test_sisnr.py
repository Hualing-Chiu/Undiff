import os
from abc import ABC, abstractmethod
from itertools import permutations

from tqdm import tqdm
import hydra.utils
import numpy as np
import torch
import torchaudio
from inference_utils import calculate_all_metrics, log_results
import logging

logging.basicConfig(filename='/home/hualing/Undiff/results/source_separation_inference/degraded.txt', level=logging.INFO, format='%(message)s')

def SiSNR(real_samples, samples):
    alpha = (samples * real_samples).sum(-1, keepdims=True) / (
            real_samples.square().sum(-1, keepdims=True) + 1e-9
    )
    real_samples_scaled = alpha * real_samples
    e_target = real_samples_scaled.square().sum(-1)
    e_res = (real_samples_scaled - samples).square().sum(-1)
    sisnr = 10 * torch.log10(e_target / (e_res + 1e-9)).cpu().numpy()
    
    return sisnr


if __name__=="__main__":
    original_path = "/home/hualing/Undiff/results/source_separation_inference/original"
    # diarization_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/diarization"
    degraded_path = "/home/hualing/Undiff/results/source_separation_inference/degraded"
    generated_path = "/home/hualing/Undiff/results/source_separation_inference/generated"

    all_sisnr = []
    sisnr_dict = {}

    files = os.listdir(original_path)
    N = len(files) // 3
    print(f"N = {N}")

    for i in tqdm(range(N)):
        original_w1, sr = torchaudio.load(os.path.join(original_path, f"Sample_{i}_1.wav"))
        original_w2, sr = torchaudio.load(os.path.join(original_path, f"Sample_{i}_2.wav"))
        original_w3, sr = torchaudio.load(os.path.join(original_path, f"Sample_{i}_3.wav"))
        generated_w1, sr = torchaudio.load(os.path.join(generated_path, f"Sample_{i}_1.wav"))
        generated_w2, sr = torchaudio.load(os.path.join(generated_path, f"Sample_{i}_2.wav"))
        generated_w3, sr = torchaudio.load(os.path.join(generated_path, f"Sample_{i}_3.wav"))
        degraded_w, sr = torchaudio.load(os.path.join(degraded_path, f"Sample_{i}.wav")) # mix 的語音

        original_w = torch.cat([original_w1, original_w2, original_w3], -1).view(1, 1, -1)
        generated_signal = [degraded_w, degraded_w, degraded_w]
        generated_ws = [torch.cat(perm, -1).view(1, 1, -1) for perm in permutations(generated_signal)]
        # print(len(generated_ws))
        # degraded_cat_w = torch.cat([degraded_w, degraded_w], -1).view(1, 1, -1)

        sisnr = max([SiSNR(original_w, generated_w) for generated_w in generated_ws])
        # sisnr = max(SiSNR(original_w, degraded_cat_w))
        logging.info(f"SI-SNR Sample_{i}.wav: {sisnr}")
        all_sisnr.append(sisnr)
        sisnr_dict[f"Sample_{i}.wav"] = sisnr
            
    all_sisnr = np.concatenate(all_sisnr)
    result_mean = np.mean(all_sisnr)
    result_std = np.std(all_sisnr)
    print(f"SiSNR mean: {result_mean}, SiSNR std: {result_std}")
    max_key = max(sisnr_dict, key=sisnr_dict.get)
    max_value = sisnr_dict[max_key]
    min_key = min(sisnr_dict, key=sisnr_dict.get)
    min_value = sisnr_dict[min_key]
    print(f"SiSNR max: {max_key}, {max_value}")
    print(f"SiSNR min: {min_key}, {min_value}")