#%%
import os
import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%%
def extract_embedding(model, path_1, path_2):
    embeddings = []

    for file in os.listdir(path_1):
        if "mic1" in file:
            signal, fs = torchaudio.load(os.path.join(path_1, file))
            i= feature_extractor(signal, return_tensors="pt", sampling_rate=fs)
            # embedding = classifier.encode_batch(waveform).squeeze().numpy()
            with torch.no_grad():
                o= model(i.input_values.squeeze(0).to('cuda'))
            embeddings.append(o.logits.squeeze().cpu().numpy())
        
    for file in os.listdir(path_2):
        if "mic1" in file:
            signal, fs = torchaudio.load(os.path.join(path_2, file))
            i= feature_extractor(signal, return_tensors="pt", sampling_rate=fs)
            # embedding = classifier.encode_batch(waveform).squeeze().numpy()
            with torch.no_grad():
                o= model(i.input_values.squeeze(0).to('cuda'))
            embeddings.append(o.logits.squeeze().cpu().numpy())
    return np.vstack(embeddings)

if __name__ == "__main__":
    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    model_name = "superb/wav2vec2-base-superb-sid"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to('cuda')
    path_1 = "/media/md01/public_datasets/VCTK-Corpus-0.92/wav16_silence_trimmed/p254"
    path_2 = "/media/md01/public_datasets/VCTK-Corpus-0.92/wav16_silence_trimmed/p248"
    # signal, fs = torchaudio.load('/media/md01/public_datasets/VCTK-Corpus-0.92/wav16_silence_trimmed/p254/p254_403_mic2.flac')
    # embeddings = classifier.encode_batch(signal)
    embeddings = extract_embedding(wav2vec, path_1, path_2)
    
    num_cluster = 2
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(embeddings)

     # 使用 PCA 將嵌入降維到 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # draw
    plt.figure(figsize=(8, 6))
    for cluster in range(num_cluster):
        cluster_points = reduced_embeddings[kmeans.labels_ == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Speaker Embeddings Clustering with KMeans')
    plt.legend()
    plt.show()        
    # print(embeddings.shape)
# %%
if __name__ == "__main__":
    # classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    model_name = "superb/wav2vec2-base-superb-sid"
    # processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

    signal_1, fs = torchaudio.load('/media/md01/public_datasets/VCTK-Corpus-0.92/wav16_silence_trimmed/p254/p254_403_mic2.flac')
    signal_2, fs = torchaudio.load('/media/md01/public_datasets/VCTK-Corpus-0.92/wav16_silence_trimmed/p254/p254_403_mic2.flac')
    signal = torch.cat([signal_1, signal_2], dim=0)
    # print(signal.shape)
    i= feature_extractor(signal, return_tensors="pt", sampling_rate=fs)
    # i = processor(signal, sampling_rate=fs, return_tensors='pt')
    print(i.input_values.shape)
    # embeddings = classifier.encode_batch(signal)
    with torch.no_grad():
        o = model(i.input_values.squeeze(0))
        # embedding = o.logits
        # embedding = embedding.mean(dim=1)
    # print(embeddings.shape)
    print(o.logits.shape) # [1, 91, 1024]
# %%
