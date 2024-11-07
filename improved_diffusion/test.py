import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchaudio
from torch.optim.adam import Adam

# from torchsummary import summary

from scipy.signal import lfilter, stft

from collections import OrderedDict

from einops import rearrange

# from torchinfo import summary

# from improved_diffusion.tasks import TaskType

class VGGM(nn.Module):
    
    def __init__(self):
        super(VGGM, self).__init__()
        # self.n_classes=n_classes
        self.features=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
            ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
            ('relu1', nn.ReLU()),
            # ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
            ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu2', nn.ReLU()),
            # ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu5', nn.ReLU()),
            # ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
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
        print(inp.shape)
        return inp


# 分割語音 and 生成 embedding
@torch.no_grad()
def generate_embeddings(waveform: torch.Tensor, sr, model, window_size=2., step_size=0.1, min_chunks=200): # min_chunks 還要改
    segment_samples = int(window_size * sr) # 每個片段長度
    step_samples = int(step_size * sr) # time step 長度

    print(segment_samples)
    print(step_samples)
    
    # use unfold
    segments = F.unfold(waveform[None], (1, segment_samples), stride=(1, step_samples))
    segments = rearrange(segments, "t s -> s t")
    # segments = waveform.unfold(1, segment_samples, step_samples)
    # total_samples = waveform.shape[1]
    print(segments.shape)
    # 最少生成3600個片段
    if segments.size(-1) < min_chunks:
        raise ValueError("numbers of segment are less than min_chunk")
    inp = preprocess(segments)
    inp = torch.from_numpy(inp).float()
    inp = rearrange(inp, "s f t -> s 1 f t")
    E = model(inp)

    # normalize embedding signal
    E_norm = nn.functional.normalize(E, dim=1)

    def fold(s, m):
        n = torch.ones_like(s)
        s = torch.einsum("ts,s->ts", s, m)
        n = torch.einsum("ts,s->ts", n, m)
        y = F.fold(s, (1, waveform.shape[-1]), (1, segment_samples), stride=(1, step_samples))
        w = F.fold(n, (1, waveform.shape[-1]), (1, segment_samples), stride=(1, step_samples))

        return (y/w).nan_to_num(0)

    return E_norm, rearrange(segments, "s t -> t s"), fold


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
    # print(nfft)
    
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
    # print(rsize, rstart)
    return mag[...,rstart:rstart+rsize]

def loss_fn(feat, A, v):
    _E = torch.einsum("Tk,kM->TM", A, v)
    loss = (_E-feat).abs().sum()
    loss += (A).abs().sum()*.2424
    loss += (v).abs().sum()*.3366
    loss += (A[1:]-A[:-1]).abs().sum()*.06
    return loss

def shrink(x:torch.Tensor, clamp_value):
    return x.sign()*F.relu(x.abs()-clamp_value)


x = torch.randn(19987)
y = torch.ones(19987)
x_seg = F.unfold(x.reshape(1,1,-1), (1,8000), stride=(1,160))
y_seg = F.unfold(y.reshape(1,1,-1), (1,8000), stride=(1,160))
print(x_seg.shape)
_x = F.fold(x_seg, (1, 19987,), (1,8000), stride=(1,160))
_y = F.fold(y_seg, (1, 19987,), (1,8000), stride=(1,160))
print(_x.shape)
print((x-(_x/_y).nan_to_num(0)).square().sum())
print(x.square().sum())

if __name__=="__main__":    
    # load model
    model = VGGM()
    # summary(model.cuda(), input_size=(1, 512, 4096))
    # model = model.load_state_dict(torch.load('./VGGM300_BEST_140_81.99.pth', map_location="cpu", weights_only=True), strict=False)
    model.load_state_dict(torch.load('/media/md01/home/hualing/Undiff/weights/VGGM300_BEST_140_81.99.pth', map_location="cpu", weights_only=True), strict=False)
    # model.eval()
    # summary(model, (1, 1, 512, 32))

    dir_path = "/media/md01/home/hualing/Undiff/results/source_separation_inference/concatenate_test"
    audio_data = []
    sr = 16000
    padding_length = sr * 1 # padding 1 sec = 16000
    for filename in os.listdir(dir_path):
        print(filename)
    # for _ in range(1):
        file_path = os.path.join(dir_path, filename)
        waveform, sr = torchaudio.load(file_path)
        # waveform, sr = torch.randn(1, 16000*4), 16000# torchaudio.load(file_path)
        # 前後 padding 1 sec
        padding_wavform = nn.functional.pad(waveform, (padding_length, padding_length), mode='constant', value=0)

        E_norm, segments, fold = generate_embeddings(padding_wavform, sr, model, 0.5, 0.01)
        print(E_norm.shape)

        T, M = E_norm.shape
        A = torch.randn(T, 2, requires_grad=True)
        v = torch.randn(2, M, requires_grad=True)

        opt_A = Adam((A,))
        opt_v = Adam((v,))

        print((E_norm).square().sum())
        print((E_norm-torch.einsum("Tk,kM->TM", A, v)).square().sum())
        for _ in range(5000):
            opt_v.zero_grad()
            loss_fn(E_norm, A, v).backward(inputs=v)
            opt_v.step()
            v.data = nn.functional.normalize(shrink(v, 1e-4), dim=1).data
            #v.data = nn.functional.normalize(v, dim=1).data

            opt_A.zero_grad()
            loss_fn(E_norm, A, v).backward(inputs=A)
            opt_A.step()
            A.data = torch.clamp(shrink(A, 1e-4), 0, 1).data
            # A.data = torch.clamp(A, 0, 1).data
        # print(A)
        print(A.argmax(-1))
        print(segments.shape)
        fold(segments, A.argmax(-1)==0)
        print((E_norm-torch.einsum("Tk,kM->TM", A, v)).square().sum())