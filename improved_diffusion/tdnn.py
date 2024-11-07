import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
signal, fs = torchaudio.load('/media/md01/public_datasets/VCTK-Corpus-0.92/wav16_silence_trimmed/p254/p254_403_mic2.flac')
embeddings = classifier.encode_batch(signal)

print(embeddings.shape)