import os
import random
import itertools
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm

import torch
import torchaudio
from torch.nn import functional as F

from improved_diffusion import bwe_utils, declipping_utils
from improved_diffusion.datasets.utils import cut_audio_segment, mel_spectrogram
from improved_diffusion.inference_utils import calculate_all_metrics, log_results
from improved_diffusion.metrics import Metric
import gc
from torch.cuda.amp import autocast
from speechbrain.inference.speaker import EncoderClassifier
# from espnet2.bin.spk_inference import Speech2Embedding
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2ForSequenceClassification
# speech2spk_embed = Speech2Embedding.from_pretrained(model_tag="espnet/voxcelebs12_ecapa_wavlm_joint")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# model_name = "superb/wav2vec2-base-superb-sid"
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
# wav2vec = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to('cuda')


class TaskType(Enum):
    BWE = auto()
    VOCODING = auto()
    DECLIPPING = auto()
    SOURCE_SEPARATION = auto()
    UNCONDITIONAL = auto()


class AbstractTask(ABC):
    def __init__(self, output_dir: str, metrics: List[Metric]):
        self.output_dir = output_dir
        self.metrics = metrics

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        raise NotImplementedError

    @abstractmethod
    def inference(self, *args, **kwargs) -> None:
        raise NotImplementedError


class UnconditionalTask(AbstractTask):
    def __init__(self, output_dir: str, metrics: List[Metric]):
        super().__init__(output_dir, metrics)

        for path in [self.generated_path]:
            if not self.exists(path):
                os.makedirs(path)

    @property
    def task_type(self) -> TaskType:
        return TaskType.UNCONDITIONAL

    @property
    def generated_path(self):
        return os.path.join(self.output_dir, "generated")

    @staticmethod
    def exists(path: str):
        return os.path.exists(path)

    def save_audios(self, pred_sample: torch.Tensor, idx: int, sr: int = 16000):
        name = f"Sample_{idx}.wav"
        torchaudio.save(
            os.path.join(self.generated_path, name), pred_sample.view(1, -1), sr
        )

    def inference(
        self,
        n_samples: int,
        model: torch.nn.Module,
        diffusion,
        target_sample_rate: int = 16000,
        segment_size: Optional[int] = None,
        device: str = "cpu",
        num_runs: int = 10
    ):
        assert self.task_type == TaskType.UNCONDITIONAL
        fake_samples = []

        # enforce default segment size as 32768
        if segment_size is None:
            segment_size = 32768

        for i in range(n_samples):
            sample = diffusion.p_sample_loop(
                model,
                (1, 1, segment_size),
                clip_denoised=False,
                model_kwargs={},
                sample_method=self.task_type,
                orig_x=None,
                progress=True,
                degradation=None,
            ).cpu()

            fake_samples.append(sample)

            self.save_audios(sample, i, sr=target_sample_rate)

            del sample
            torch.cuda.empty_cache()

        scores = calculate_all_metrics(fake_samples, self.metrics, reference_wavs=None)
        log_results(results_dir=self.output_dir, res=scores)


class BaseInverseTask(UnconditionalTask):
    def __init__(self, output_dir: str, metrics: List[Metric]):
        super().__init__(output_dir, metrics)

        for path in [self.generated_path, self.original_path, self.degraded_path, self.concatenate_path, self.diarization_path]:
            if not self.exists(path):
                os.makedirs(path)

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        raise NotImplementedError

    @property
    def generated_path(self):
        return os.path.join(self.output_dir, "generated")

    @property
    def original_path(self):
        return os.path.join(self.output_dir, "original")

    @property
    def degraded_path(self):
        return os.path.join(self.output_dir, "degraded")

    # my own concatenate function
    @property
    def concatenate_path(self):
        return os.path.join(self.output_dir, "concatenate")
    
    @property
    def diarization_path(self):
        return os.path.join(self.output_dir, "diarization")

    @staticmethod
    def exists(path: str):
        return os.path.exists(path)

    def prepare_data(self, audio_files: List[str]) -> Dict:
        return {"files": audio_files}

    def prepare_audio_before_degradation(self, x: List[torch.Tensor]) -> torch.Tensor:
        return x[0]

    @staticmethod
    def load_audio(
        path: str,
        target_sample_rate: int = 16000,
        segment_size: Optional[int] = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        print(path)
        x, sr = torchaudio.load(path)
        x = torchaudio.functional.resample(x, sr, target_sample_rate)
        if segment_size is not None:
            x = cut_audio_segment(x, segment_size)
        x = torchaudio.functional.vad(x, target_sample_rate)
        x = x.to(device).unsqueeze(0)
        return x

    def load_audios(self, paths: Tuple[str], *args, **kwargs) -> List[torch.Tensor]:
        return [self.load_audio(p, *args, **kwargs) for p in paths]

    def save_audios(
        self,
        pred_sample: torch.Tensor,
        degraded_sample: torch.Tensor,
        original_sample: torch.Tensor,
        idx: int,
        sr: int = 16000,
    ):
        name = f"Sample_{idx}.wav"
        torchaudio.save(
            os.path.join(self.generated_path, name), pred_sample.view(1, -1), sr
        )
        torchaudio.save(
            os.path.join(self.degraded_path, name), degraded_sample.view(1, -1), sr
        )
        torchaudio.save(
            os.path.join(self.original_path, name), original_sample.view(1, -1), sr
        )

    @abstractmethod
    def degradation(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inference(
        self,
        audio_files: List[str],
        model: torch.nn.Module,
        diffusion,
        target_sample_rate: int = 16000,
        segment_size: Optional[int] = None,
        device: str = "cpu",
        num_runs: int = 10
    ):
        assert self.task_type != TaskType.UNCONDITIONAL  # inverse tasks only
        files_dict = self.prepare_data(audio_files)

        fake_samples = []
        real_samples = []

        for i, f in enumerate(zip(*files_dict.values())):
            x = self.load_audios(f, target_sample_rate, segment_size, device)
            x = self.prepare_audio_before_degradation(x)
            print(x.shape)

            degraded_sample = self.degradation(x).cpu()
            sample = diffusion.p_sample_loop(
                model,
                x.shape,
                clip_denoised=False,
                model_kwargs={},
                sample_method=self.task_type,
                orig_x=x,
                progress=True,
                degradation=self.degradation,
                task_kwargs=None
            ).cpu()

            x = x.cpu()
            real_samples.append(x)
            fake_samples.append(sample)

            self.save_audios(sample, degraded_sample, x, i, sr=target_sample_rate)

            del sample, x, degraded_sample
            torch.cuda.empty_cache()

        scores = calculate_all_metrics(
            fake_samples, self.metrics, reference_wavs=real_samples
        )
        log_results(results_dir=self.output_dir, res=scores)


class BWETask(BaseInverseTask):
    @property
    def task_type(self) -> TaskType:
        return TaskType.BWE

    def degradation(self, x: torch.Tensor) -> torch.Tensor:
        lp_filter = bwe_utils.get_FIR_lowpass(order=200, fc=2000, beta=1, sr=16000)
        return bwe_utils.apply_low_pass_firwin(x, lp_filter)


class DeclippingTask(BaseInverseTask):
    clip_value = None

    @property
    def task_type(self) -> TaskType:
        return TaskType.DECLIPPING

    def degradation(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_value is None:
            self.clip_value = declipping_utils.get_clip_value_from_SDR(x, 3.0)
        return torch.clip(x, min=-self.clip_value, max=self.clip_value)


class VocodingTask(BaseInverseTask):
    @property
    def task_type(self) -> TaskType:
        return TaskType.VOCODING

    def degradation(self, x: torch.Tensor) -> torch.Tensor:
        return mel_spectrogram(
            x.squeeze(1),
            n_fft=1024,
            num_mels=80,
            sampling_rate=16000,
            win_size=1024,
            hop_size=256,
            fmin=0,
            fmax=8000,
            use_log_normalize=False,
        )

    def save_audios(
        self,
        pred_sample: torch.Tensor,
        degraded_sample: torch.Tensor,
        original_sample: torch.Tensor,
        idx: int,
        sr: int = 16000,
    ):
        name = f"Sample_{idx}.wav"
        torchaudio.save(
            os.path.join(self.generated_path, name), pred_sample.view(1, -1), sr
        )
        torch.save(
            degraded_sample, os.path.join(self.degraded_path, name)
        )  # save mel spec tensor
        torchaudio.save(
            os.path.join(self.original_path, name), original_sample.view(1, -1), sr
        )


class SourceSeparationTask(BaseInverseTask):
    @property
    def task_type(self) -> TaskType:
        return TaskType.SOURCE_SEPARATION

    def prepare_data(self, audio_files: List[str]):
        # filtered_mic2_audio_files = [[file for file in files if "mic1" in file] for files in audio_files]
        filtered_audio_files = [[file for file in files if file.endswith('wav')] for files in audio_files]
        n_samples = min([len(files) for files in filtered_audio_files])

        return {f"spk{i}": random.sample(files, k=n_samples)  for i, files in enumerate(filtered_mic2_audio_files)}

    def prepare_audio_before_degradation(self, x: List[torch.Tensor]) -> torch.Tensor:
        min_sample_length = min(map(lambda tensor: tensor.size(-1), x))
        truncated_x = list(map(lambda tensor: tensor[..., :min_sample_length], x))
        return torch.cat(truncated_x, dim=0) # dim=-1 to dim=0
        # return torch.cat(truncated_x, dim=-1)

    def save_audios(
        self,
        pred_sample: torch.Tensor,
        degraded_sample: torch.Tensor,
        original_sample: torch.Tensor,
        idx: int,
        n_spk: int,
        sr: int = 16000,
    ):
        pred_chunked = torch.chunk(
            pred_sample, chunks=n_spk, dim=0 # dim=0 -> batch # modify
        )  # explicit number of chunks 2
        orig_chunked = torch.chunk(
            original_sample, chunks=n_spk, dim=0
        )  # explicit number of chunks 2
        for i, (cur_pred, cur_orig) in enumerate(zip(pred_chunked, orig_chunked)):
            name = f"Sample_{idx}_{i + 1}.wav"
            torchaudio.save(
                os.path.join(self.generated_path, name), cur_pred.view(1, -1), sr
            )
            torchaudio.save(
                os.path.join(self.original_path, name), cur_orig.view(1, -1), sr
            )
        
        # concate the separate audio
        concatenated_pred = torch.cat(pred_chunked, dim=-1)
        name = f"Sample_{idx}.wav"
        torchaudio.save(
            os.path.join(self.concatenate_path, name), concatenated_pred.view(1, -1), sr
        )

        # redefine name for degraded
        name = f"Sample_{idx}.wav"
        torchaudio.save(
            os.path.join(self.degraded_path, name), degraded_sample.view(1, -1), sr
        )

    # def degradation(self, x: torch.Tensor) -> torch.Tensor:
    #     return x[: x.size(0) // 2, :, :] + x[x.size(0) // 2 :, :, :]
        # return x[:, :, : x.size(-1) // 2] + x[:, :, x.size(-1) // 2 :]
    def degradation(self, x: torch.Tensor) -> torch.Tensor:
         return torch.stack([s for s in torch.chunk(x, 2, dim=0)]).sum(0)

    def inference(
        self,
        audio_files: List[str],
        model: torch.nn.Module,
        diffusion,
        target_sample_rate: int = 16000,
        segment_size: Optional[int] = None,
        device: str = "cpu",
        num_runs: int = 5
    ):
        assert self.task_type != TaskType.UNCONDITIONAL  # inverse tasks only
        files_dict = self.prepare_data(audio_files)

        fake_samples = []
        real_samples = []
        files_key = list(files_dict.keys())

        for i, f in enumerate(zip(*files_dict.values())): # f 是路徑tuple
            x = self.load_audios(f, target_sample_rate, segment_size, device)
            x = self.prepare_audio_before_degradation(x)
            print(x.shape)
            degraded_sample = self.degradation(x).cpu()

            reference_1 = random.choice([file for file in files_dict[files_key[0]] if file not in f[0]])
            reference_2 = random.choice([file for file in files_dict[files_key[1]] if file not in f[1]])
            reference_f = (reference_1, reference_2)
            r_x = self.load_audios(reference_f, target_sample_rate, segment_size, device)
            r_x = self.prepare_audio_before_degradation(r_x)

            with torch.no_grad():
                r_embeddings = classifier.encode_batch(r_x.squeeze(1))

            # multi sample
            sample_list = []
            for _ in tqdm(range(num_runs)):
                sample = diffusion.p_sample_loop(
                    model,
                    x.shape,
                    clip_denoised=False,
                    model_kwargs={},
                    sample_method=self.task_type,
                    orig_x=x,
                    progress=True,
                    degradation=self.degradation,
                    task_kwargs= {'r_e': r_embeddings}
                ).cpu()

                sample_list.append(sample)
                del sample
                torch.cuda.empty_cache()
                gc.collect()

            x = x.cpu()
            real_samples.append(x)
            # samples_tensor = torch.stack(sample_list, dim=0)
            # batch permutation
            B = x.shape[0] # batch num = speaker num
            samples_sum = sample_list[0].clone() # (B, C, T)
            for j in range(1, num_runs):
                sample_next = sample_list[j]
                base_sample = sample_list[0]
                best_perm = None
                best_score = float("-inf")
                for perm in itertools.permutations(range(B)):
                    reordered_sample = sample_next[list(perm)]
                    sisnr_score = sum(self.sisnr(base_sample[k], reordered_sample[k]) for k in range(B))
                
                    if sisnr_score > best_score:
                        best_score = sisnr_score
                        best_perm = perm

                sample_next = sample_next[list(best_perm)]
                # A1, A2 = sample_list[0][0], sample_list[0][1]
                # B1, B2 = sample_next[0], sample_next[1]

                # cal sisnr
                # score_A1_B1 = self.sisnr(A1, B1)
                # score_A1_B2 = self.sisnr(A1, B2)

                # if score_A1_B2 > score_A1_B1:
                #     B1, B2 = B2, B1

                # sample_next = torch.stack([B1, B2], dim=0)
                samples_sum += sample_next
                del sample_next
                torch.cuda.empty_cache()
                gc.collect()

            samples_mean = samples_sum / num_runs
            # mean_sample = samples_tensor.mean(dim=0)
            # print(f"mean_sample: {mean_sample.shape}")
            fake_samples.append(samples_mean)
            self.save_audios(samples_mean, degraded_sample, x, i, len(audio_files), sr=target_sample_rate)

            # del sample, x, degraded_sample, reference_1, reference_2, reference_f, r_x, r_embeddings
            del sample_list, samples_mean, x, degraded_sample
            torch.cuda.empty_cache()
            gc.collect()

        scores = calculate_all_metrics(
            fake_samples, self.metrics, reference_wavs=real_samples
        )
        log_results(results_dir=self.output_dir, res=scores)

    def sisnr(self, x, y):
        alpha = (x * y).sum(-1, keepdims=True) / (
            x.square().sum(-1, keepdims=True) + 1e-9
        )
        real_samples_scaled = alpha * x
        e_target = real_samples_scaled.square().sum(-1)
        e_res = (real_samples_scaled - y).square().sum(-1)
        return 10 * torch.log10(e_target / (e_res + 1e-9)).cpu().numpy()