import os
import torch
import torchaudio
import random
import json
from collections import defaultdict
from tqdm import tqdm

def get_speakers_wav(audio_dir):
    speaker_wavs = defaultdict(list)

    for root, _, files in os.walk(audio_dir):
        if os.path.basename(root) == "mix":
            continue
        
        for file in files:
            if file.endswith(".wav") and "mic2" not in file:
                full_path = os.path.join(root, file)
                speaker_id = os.path.basename(os.path.dirname(full_path))
                speaker_wavs[speaker_id].append(full_path)

    return speaker_wavs

def cal_mean_variance(audio_dir, sample_rate=None):
    speaker_wavs = get_speakers_wav(audio_dir)
    all_speakers = list(speaker_wavs.keys())
    # if len(all_speakers) < num_speakers:
    #     raise ValueError(f"Not enough speakers in the directory. Found {len(all_speakers)}, expected {num_speakers}.")
    

    total_sum = 0.0
    total_squared_sum = 0.0
    total_sample = 0

    # pick 10 speakers
    # speaker_dirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
    # selected_speakers = random.sample(all_speakers, num_speakers)
    # print(f"Selected speakers: {selected_speakers}")

    # for speaker in selected_speakers:
    for speaker in all_speakers:
        for file_path in tqdm(speaker_wavs[speaker], desc=f"Processing {speaker}"):
            waveform, sr = torchaudio.load(file_path)
            
            # Resample if necessary
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                waveform = resampler(waveform)
            
            # Calculate mean and variance
            waveform = waveform.mean(dim=0)
            total_sum += waveform.sum().item()
            total_squared_sum += (waveform ** 2).sum().item()
            total_sample += waveform.numel()

    mean = total_sum / total_sample
    variance = (total_squared_sum / total_sample) - (mean ** 2)
    return mean, variance

if __name__ == "__main__":
    # audio_dir = "/media/md01/public_datasets/VCTK-Corpus-0.92/wav16_silence_trimmed"  # Replace with your audio directory
    # audio_dir = "/media/md01/public_datasets/LibriTTS_R/train-clean-100"  # Replace with your audio directory
    # audio_dir = "/home/public_datasets/VCTK-Corpus-0.92/wav48_silence_trimmed"  # Replace with your audio directory
    audio_dir = "/media/md01/home/hualing/data/wsj0-mix/2speakers/wav16k/min"  # Replace with your audio directory
    mean, variance = cal_mean_variance(audio_dir, sample_rate=16000)
    print(f"Mean: {mean}, Variance: {variance}")

    # Save the mean and variance to a JSON file
    stats = {
        "mean": mean,
        "variance": variance
    }
    # jsonfile = os.path.join("/home/hualing/Undiff/improved_diffusion", "mean_variance.json")
    with open("wsj0_mean_variance.json", "w") as f:
        json.dump(stats, f, indent=4)