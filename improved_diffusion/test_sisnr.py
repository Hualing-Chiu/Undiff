import os
from abc import ABC, abstractmethod

from tqdm import tqdm
import hydra.utils
import numpy as np
import torch
import torchaudio

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
    original_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/original"
    diarization_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/diarization"
    degraded_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/degraded"
    generated_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/generated"

    all_sisnr = []

    files = os.listdir(original_path)
    N = len(files) // 2
    print(f"N = {N}")

    for i in tqdm(range(N)):
        original_w1, sr = torchaudio.load(os.path.join(original_path, f"Sample_{i}_1.wav"))
        original_w2, sr = torchaudio.load(os.path.join(original_path, f"Sample_{i}_2.wav"))
        diarization_w1, sr = torchaudio.load(os.path.join(generated_path, f"Sample_{i}_1.wav"))
        diarization_w2, sr = torchaudio.load(os.path.join(generated_path, f"Sample_{i}_2.wav"))
        # degraded_w, sr = torchaudio.load(os.path.join(degraded_path, f"Sample_{i}.wav")) # mix 的語音

        original_w = torch.cat([original_w1, original_w2], -1).view(1, 1, -1)
        diarization_ws = [torch.cat([diarization_w1, diarization_w2], -1).view(1, 1, -1),
                        torch.cat([diarization_w2, diarization_w1], -1).view(1, 1, -1)]
        # degraded_cat_w = torch.cat([degraded_w, degraded_w], -1).view(1, 1, -1)

        sisnr = max([SiSNR(original_w, diarization_w) for diarization_w in diarization_ws])
        # sisnr = max(SiSNR(original_w, degraded_cat_w))
        all_sisnr.append(sisnr)
            
    all_sisnr = np.concatenate(all_sisnr)
    result_mean = np.mean(all_sisnr)
    result_std = np.std(all_sisnr)
    print(f"SiSNR mean: {result_mean}, SiSNR std: {result_std}")
