# %%
import os
import numpy as np
from numpy.fft import fft
import math

import torch
import torchaudio
import matplotlib.pyplot as plt

if __name__=="__main__":
    generated_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/test_original"

    n_fft = 1024
    n_mels = 128
    hop_length = 512
    win_length = 1024


    files = os.listdir(generated_path)
    N = len(files) // 2
    print(f"N = {N}")

    for i in range(N):
        diarization_w1, sr = torchaudio.load(os.path.join(generated_path, f"Sample_{i}_1.wav"))
        diarization_w2, sr = torchaudio.load(os.path.join(generated_path, f"Sample_{i}_2.wav"))

        # temp = diarization_w1[..., :diarization_w1.size(-1) // 2]
        cat_wv = torch.cat([diarization_w1[..., :diarization_w1.size(-1) // 2], diarization_w2[..., diarization_w2.size(-1) // 2:]], -1)
        # cat_wv1 = torch.cat([diarization_w1, diarization_w1], -1).view(1, 1, -1)

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = sr,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length
        )

        # waveform to mel spectrogram
        mel_spec_1 = mel_spectrogram(diarization_w1)
        mel_spec_2 = mel_spectrogram(cat_wv)

        # print shape [channels, mel_bin]
        print(mel_spec_1.shape)
        print(mel_spec_2.shape)

        T = mel_spec_1.size(-1)
        print(f"T = {T}")
        diff_1 = (mel_spec_1[...,:-1] - mel_spec_1[...,1:]).abs().mean(-1)
        diff_2 = (mel_spec_1[...,:-1] - mel_spec_2[...,1:]).abs().mean(-1)
        
        print(diff_1 < diff_2)
        # 畫出 Mel spectrogram
        # plt.figure(figsize=(10, 4))
        # plt.imshow(mel_spec_1[0].log().numpy(), cmap='viridis', origin='lower', aspect='auto')
        # plt.colorbar(format="%+2.0f dB")
        # plt.title('Mel Spectrogram')
        # plt.ylabel('Mel Frequency Bins')
        # plt.xlabel('Time Frames')
        # plt.show()

        # plt.figure(figsize=(10, 4))
        # plt.imshow(mel_spec_2[0].log().numpy(), cmap='viridis', origin='lower', aspect='auto')
        # plt.colorbar(format="%+2.0f dB")
        # plt.title('Mel Spectrogram')
        # plt.ylabel('Mel Frequency Bins')
        # plt.xlabel('Time Frames')
        # plt.show()

# %%
