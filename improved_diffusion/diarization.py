import os
import numpy as np
from numpy.fft import fft
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from torchsummary import summary
from torch.optim.adam import Adam

from collections import OrderedDict

from scipy.signal import lfilter, stft

from einops import rearrange
# from improved_diffusion.tasks import TaskType

class VGGM(nn.Module):
    
    def __init__(self):
        super(VGGM, self).__init__()
        # self.n_classes=n_classes
        self.features=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
            ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
            ('relu1', nn.ReLU()),
            ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
            ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu2', nn.ReLU()),
            ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu5', nn.ReLU()),
            ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
            ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=(1,1))),
            ('bn6', nn.BatchNorm2d(4096, momentum=0.5)),
            ('relu6', nn.ReLU()),
            ('apool6', nn.AdaptiveAvgPool2d((1,1))),
            ('flatten', nn.Flatten())]))
            
        # self.classifier=nn.Sequential(OrderedDict([
        #     ('fc7', nn.Linear(4096, 1024)),
        #     #('drop1', nn.Dropout()),
        #     ('relu7', nn.ReLU()),
        #     ('fc8', nn.Linear(1024, n_classes))]))
    
    def forward(self, inp):
        inp=self.features(inp)
        #inp=inp.view(inp.size()[0],-1)
        # inp=self.classifier(inp)
        return inp

def rm_dc_n_dither(audio):
    # All files 16kHz tested..... Will copy for 8kHz from author's matlab code later
    alpha=0.99
    b=[1,-1]
    a=[1,-alpha]
    
    audio=lfilter(b,a,audio)
    
    dither=np.random.uniform(low=-1,high=1, size=audio.shape)
    spow=np.std(audio)
    return audio+(1e-6*spow)*dither

def preemphasis(audio, alpha=0.97):
    b=[1, -alpha]
    a=1
    return lfilter(b, a, audio)

def normalize_frames(m,epsilon=1e-12):
    return (m-m.mean(1, keepdims=True))/np.clip(m.std(1, keepdims=True),epsilon, None)

def preprocess(audio, buckets=None, sr=16000, Ws=25, Ss=10, alpha=0.97):
    #ms to number of frames
    if not buckets:
        buckets={100: 2,
             200: 5,
             300: 8,
             400: 11,
             500: 14,
             600: 17,
             700: 20,
             800: 23,
             900: 27,
             1000: 30}
    
    Nw=round((Ws*sr)/1000)
    Ns=round((Ss*sr)/1000)
    
    
    #hamming window func signature
    window=np.hamming
    #get next power of 2 greater than or equal to current Nw
    nfft=1<<(Nw-1).bit_length()
    
    # Remove DC and add small dither
    audio=rm_dc_n_dither(audio)
    
    # Preemphasis filtering
    audio=preemphasis(audio, alpha)
    
    #get 512x300 spectrograms
    _, _, mag=stft(audio,
    fs=sr, 
    window=window(Nw), 
    nperseg=Nw, 
    noverlap=Nw-Ns, 
    nfft=nfft, 
    return_onesided=False, 
    padded=False, 
    boundary=None)

    mag=normalize_frames(np.abs(mag))
    
    #Get the largest bucket smaller than number of column vectors i.e. frames
    rsize=max(i for i in buckets if i<=mag.shape[1])
    rstart=(mag.shape[1]-rsize)//2
    #Return truncated spectrograms
    return mag[..., rstart:rstart+rsize]

# 分割語音 and 生成 embedding
@torch.no_grad()
def generate_embeddings(waveform: torch.Tensor, sr, model, device, window_size=1, step_size=0.05, min_chunks=200): # min_chunks 還要改
    segment_samples = int(window_size * sr) # 每個 segment 長度
    step_samples = int(step_size * sr) # time step 長度

    # print(segment_samples)
    # print(step_samples)
    
    # use unfold
    segments = F.unfold(waveform[None], (1, segment_samples), stride=(1, step_samples)) 
    segments = rearrange(segments, "t s -> s t") # [45, 16000] [num_segment, length of one segment]
    # total_samples = waveform.shape[1]
    print(waveform.shape)
    print(segments.shape)

    # 最少生成3600個片段
    # if segments.size(-1) < min_chunks:
    #     raise ValueError("numbers of segment are less than min_chunk")
    
    # waveform -> spectrogram
    inp = preprocess(segments)
    inp = torch.from_numpy(inp).float().to(device) # np array to tensor
    # print(inp.shape)

    E = model(inp[:, None])

    # normalize embedding signal
    E_norm = F.normalize(E, dim=1)

    def permutation(segments, A_mask):
        S = torch.einsum("ts,s->ts", segments, A_mask) # [t * s]
        S_fold = F.fold(S, (1, waveform.shape[-1]), (1, segment_samples), stride=(1, step_samples))

        # copy one tensor
        N = torch.ones_like(S)
        N = torch.einsum("ts,s->ts", N, A_mask)
        N_fold = F.fold(N, (1, waveform.shape[-1]), (1, segment_samples), stride=(1, step_samples))

        return (S_fold / N_fold).nan_to_num(0)
    
    
    return E_norm, rearrange(segments, "s t -> t s"), permutation

def compute_loss(E_norm, A, Psi, K, T):
    E_cal = torch.einsum("TK,KM->TM", A, Psi) # A * Psi
    loss = (E_cal - E_norm).abs().sum()
    loss += Psi.abs().sum() * 0.3366
    loss += A.abs().sum() * 0.2424
    J = (A[1:, :] - A[-1, :]).abs().sum() / (K * T) # jitter loss
    loss += J * 0.06
    return loss
    
def shrink(X: torch.Tensor, value):
    return X.sign() * nn.functional.relu(X.abs() - value)

def project_unitdisk(X: torch.Tensor):
    L2_norm = torch.norm(X, p=2, dim=1, keepdim=True)
    return X / L2_norm



if __name__=="__main__":    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = VGGM()
    # summary(model.cuda(), input_size=(1, 512, 4096))
    model.load_state_dict(torch.load('/media/md01/home/hualing/Undiff/weights/VGGM300_BEST_140_81.99.pth', map_location=device), strict=False)
    model.to(device)
    # model.eval()

    dir_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/concatenate"
    audio_data = []
    sr = 16000
    padding_length = int(sr * 0.5) # padding 0.5 sec = 8000
    buckets={100: 2,
        200: 5,
        300: 8,
        400: 11,
        500: 14,
        600: 17,
        700: 20,
        800: 23,
        900: 27,
        1000: 30}

    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        waveform, sr = torchaudio.load(file_path)
        print(filename)
        # 前後 padding 1 sec
        padding_wavform = F.pad(waveform, (padding_length, padding_length), mode='constant', value=0)

        E_norm, segments, permutation = generate_embeddings(padding_wavform, sr, model, device) # (T × M)
        segments = segments.to(device)
        # print(E_norm.shape)

        # random two matrix Psi & A
        T, M = E_norm.shape
        K = 2 # number of speaker

        A = torch.randn(T, K, requires_grad=True, device=device)
        Psi = torch.randn(K, M, requires_grad=True, device=device)

        # define opt
        opt_A = Adam([A],)
        opt_Psi = Adam([Psi],)

        # print(E_norm)
        # print(E_norm - torch.einsum("TK,KM->TM", A, Psi)) # 矩陣相乘
        i = 0
        while i < 5000:
            # compute loss
            L = compute_loss(E_norm, A, Psi, K, T)
            # calculate gradient of Psi
            opt_Psi.zero_grad()
            L.backward(inputs=Psi)
            opt_Psi.step()
            # Psi.data = nn.functional.normalize(shrink(Psi), dim=1).data
            Psi.data = shrink(Psi.data, 1e-4)
            Psi.data = project_unitdisk(Psi.data)

            # recompute loss
            L = compute_loss(E_norm, A, Psi, K, T)
            #calculate gradient of A
            opt_A.zero_grad()
            L.backward(inputs=A)
            opt_A.step()
            A.data = torch.clamp(shrink(A.data, 1e-4), 0, 1).data
            i = i + 1

        # print(E_norm - torch.einsum("TK,KM->TM", A, Psi))
        # print(A.argmax(-1))
        # print(A.shape) # [45, 2]

        # print(segments.shape)
        new_waveform_list = []
        for k in range(0, K):
            new_waveform = permutation(segments, (A.argmax(-1) == k))
            # print(new_waveform.shape)
            new_waveform = new_waveform.squeeze(1)
            new_waveform = new_waveform[..., padding_length : -padding_length]
            # print(new_waveform.shape)
            chunks = torch.chunk(new_waveform, K, dim=-1)
            S, T = chunks[0].shape
            new_chunks = torch.zeros(S, T).to(device)

            for chunk in chunks:
                new_chunks.data += chunk.data
            
            # print(new_chunks.shape)
            new_waveform_list.append(new_chunks)

        new_path = '/media/md01/home/hualing/Undiff/results/source_separation_inference/diarization'
        
        if not os.path.exists(new_path):
            os.makedirs(new_path)

        for i, _waveform in enumerate(new_waveform_list):
            name = filename[:-4] + f"_{i + 1}.wav"
            torchaudio.save(
                os.path.join(new_path, name), _waveform.view(1, -1).cpu(), sr
            )