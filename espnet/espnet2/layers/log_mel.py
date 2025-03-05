from typing import Tuple

import librosa
import numpy as np
import pywt
import scipy
import torch
import torch.nn.functional as F
import torchaudio.transforms as transforms
from nnAudio import Spectrogram
from spafe.fbanks import bark_fbanks, gammatone_fbanks
import torch.nn as nn
import math


class Cqt(torch.nn.Module):
    """Convert Raw audio to CQT"""

    def __init__(
        self,
        fs: int = 16000,
        n_bins: int = 80,
        hop_length: int = 160,
        bins_per_octave: int = 12,
    ):
        super().__init__()

        # We use 12 bins per octave 
        self.bins_per_octave=bins_per_octave
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.fs = fs
        self.extract_cqt = Spectrogram.CQT(
            sr=self.fs, n_bins=self.n_bins, bins_per_octave=self.bins_per_octave, hop_length=self.hop_length
        )

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        #
        output = []
        output_lens = []
        for i, instance in enumerate(feat):
            if len(feat[i])<8193:
                pad_length = 8193 - len(feat[i])
                a = F.pad( feat[i], (0, pad_length), mode="constant")
                cqt = self.extract_cqt(a.unsqueeze(0)).squeeze().T
            else:
                cqt = self.extract_cqt(feat[i].unsqueeze(0)).squeeze().T
            output.append(cqt)
            output_lens.append(cqt.shape[0])
        # Hardcode again
        cqt_feat = torch.stack(output, 0)

        output_lens = cqt_feat.new_full(
            [cqt_feat.size(0)], fill_value=cqt_feat.size(1), dtype=torch.long
        )
        return cqt_feat, output_lens.cuda()


class Mfcc(torch.nn.Module):
    """Convert Raw audio to MFCC
    #Mfcc

    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        window_size: int = 512,
        hop_length: int = 160,
    ):
        super().__init__()
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.mfcc_transform = transforms.MFCC(
            sample_rate=self.fs,
            n_mfcc=self.n_mels,
            melkwargs={
                "n_fft": self.n_fft,
                "win_length": self.window_size,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
            },
        )

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: Raw audio waveform
        # (B, T) -> (B, D, n_mels)
        output = []
        output_lens = []

        for instance in feat:
            mfcc = self.mfcc_transform(instance)
            output.append(mfcc.T)
            output_lens.append(mfcc.size(1))

        mfcc_feat = torch.stack(output, 0).cuda()

        output_lens = mfcc_feat.new_full(
            [mfcc_feat.size(0)], fill_value=mfcc_feat.size(1), dtype=torch.long
        )

        return mfcc_feat, output_lens


#
#
class Gamma(torch.nn.Module):
    """Convert Raw audio to GAMMA"""

    def __init__(
        self,
        fs: int = 16000,
        n_filts: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window_size: int = 400,
    ):
        super().__init__()
        self.fs = fs
        self.n_filts = n_filts
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.gammatone_filter_bank = gammatone_fbanks.gammatone_filter_banks(
            nfilts=self.n_filts, nfft=self.n_fft, fs=self.fs
        )[0]

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        def extract_gamma(wav):
            magnitude = (
                np.abs(
                    librosa.stft(
                        y=wav,
                        win_length=self.window_size,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                    )
                )
                ** 2
            )
            Gam = np.dot(self.gammatone_filter_bank, magnitude)
            LogGamSpec = librosa.power_to_db(Gam, ref=np.max)
            return LogGamSpec.T

        output = []
        output_lens = []
        for i, instance in enumerate(feat):
            gamma = extract_gamma(feat[i].cpu().numpy())
            output.append(torch.Tensor(gamma))
            output_lens.append(gamma.shape[0])
        # Hardcode again
        gamma_feat = torch.stack(output, 0).cuda()
        output_lens = gamma_feat.new_full(
            [gamma_feat.size(0)], fill_value=gamma_feat.size(1), dtype=torch.long
        )
        return gamma_feat, output_lens.cuda()


class LogMel(torch.nn.Module):
    """Convert Raw audio to LOGMEL using torchaudio"""

    def __init__(
        self,
        fs: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window_size: int = 512,
        fmin: float = 0.0,
        fmax: float = None,
    ):
        super().__init__()
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else fs // 2
        self.window_size = window_size
        self.mel_spectrogram_transform = transforms.MelSpectrogram(
            win_length=self.window_size,
            sample_rate=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        self.amplitude_to_db_transform = transforms.AmplitudeToDB()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = []
        output_lens = []

        for instance in feat:
            mel_spectrogram = self.mel_spectrogram_transform(instance)
            log_mel_spectrogram = self.amplitude_to_db_transform(mel_spectrogram)
            output.append(log_mel_spectrogram.T)
            output_lens.append(log_mel_spectrogram.size(1))

        logmel_feat = torch.stack(output, 0).cuda()

        output_lens = logmel_feat.new_full(
            [logmel_feat.size(0)], fill_value=logmel_feat.size(1), dtype=torch.long
        )

        return logmel_feat, torch.tensor(output_lens).cuda()


class Bark(torch.nn.Module):
    def __init__(
        self,
        fs: int = 16000,
        n_filts: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window_size: int = 400,
    ):
        super().__init__()
        self.fs = fs
        self.n_filts = n_filts
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.bark_filters = bark_fbanks.bark_filter_banks(
            nfilts=self.n_filts, nfft=n_fft, fs=self.fs
        )[0]

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def extract_bark(wav):
            magnitude = (
                np.abs(librosa.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length))
                ** 2
            )
            # Compute magnitude spectrogram
            S_bark = np.dot(self.bark_filters, magnitude)
            S_bark = librosa.amplitude_to_db(S_bark, ref=np.max)
            return S_bark.T

        output = []
        output_lens = []
        for i, instance in enumerate(feat):
            bark = extract_bark(feat[i].cpu().numpy())
            output.append(torch.Tensor(bark))
            output_lens.append(bark.shape[0])
        # Hardcode again
        bark_feat = torch.stack(output, 0).cuda()
        output_lens = bark_feat.new_full(
            [bark_feat.size(0)], fill_value=bark_feat.size(1), dtype=torch.long
        )
        return bark_feat, output_lens.cuda()


class GroupDelay(nn.Module):
    """Convert Raw audio to Group Delay Features"""

    def __init__(
        self,
        fs: int = 16000,
        n_filts: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window_size: int = 400,
    ):
        super().__init__()
        self.fs = fs
        self.n_filts = n_filts
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size

        # Create Mel filterbank as a PyTorch tensor
        mel_filter_bank = librosa.filters.mel(
            sr=self.fs, n_fft=self.n_fft, n_mels=self.n_filts
        )
        self.mel_filter_bank = torch.from_numpy(mel_filter_bank).float()

        # Create window function as a PyTorch tensor
        window = scipy.signal.windows.chebwin(self.window_size, at=30)
        self.window = torch.from_numpy(window).float()

    def preemphasis(self, signal, coeff=0.97):
        """Apply pre-emphasis to the input signal."""
        return torch.cat([signal[:, :1], signal[:, 1:] - coeff * signal[:, :-1]], dim=1)

    def frame_signal(self, signal, frame_length, frame_step):
        """Frame the signal into overlapping frames."""
        # signal: (batch_size, signal_length)
        batch_size, signal_length = signal.size()
        num_frames = 1 + (signal_length - frame_length) // frame_step

        indices = torch.arange(0, frame_length, device=signal.device).unsqueeze(
            0
        ).expand(num_frames, -1) + torch.arange(
            0, num_frames * frame_step, frame_step, device=signal.device
        ).unsqueeze(1)  # Shape: (num_frames, frame_length)

        frames = signal[:, indices]  # Shape: (batch_size, num_frames, frame_length)
        return frames

    def compute_lpc(self, frames, order):
        """Compute LPC coefficients using autocorrelation method."""
        # frames: (batch_size, num_frames, frame_length)
        batch_size, num_frames, frame_length = frames.size()
        # Zero-pad frames to next power of 2 for FFT
        n_fft = 1 << (frame_length - 1).bit_length()
        frames_padded = F.pad(frames, (0, n_fft - frame_length))

        # Compute FFT
        frames_fft = torch.fft.rfft(frames_padded, n=n_fft, dim=2)
        # Compute power spectrum
        power_spectrum = frames_fft.abs() ** 2
        # Compute autocorrelation via IFFT
        autocorr = torch.fft.irfft(power_spectrum, n=n_fft, dim=2)
        # Keep only first (order + 1) coefficients
        autocorr = autocorr[:, :, : order + 1]

        # Levinson-Durbin recursion to compute LPC coefficients
        lpc_coeffs = self.levinson_durbin(
            autocorr, order
        )  # Shape: (batch_size, num_frames, order)
        return lpc_coeffs

    def levinson_durbin(self, r, order):
        """Levinson-Durbin recursion for batch data."""
        # r: (batch_size, num_frames, order + 1)
        batch_size, num_frames, _ = r.size()
        a = torch.zeros(batch_size, num_frames, order + 1, device=r.device)
        e = r[:, :, 0]
        a[:, :, 0] = 1.0

        for i in range(1, order + 1):
            # Use torch.flip instead of negative step slicing
            reversed_a = torch.flip(a[:, :, :i], dims=[2])
            acc = torch.sum(reversed_a * r[:, :, 1 : i + 1], dim=2)

            # Add small epsilon to prevent division by zero
            k = -acc / (e + 1e-10)

            a_new = torch.cat(
                [a[:, :, :i], torch.zeros(batch_size, num_frames, 1, device=r.device)],
                dim=2,
            )
            a_new[:, :, i] = k

            # Use torch.flip for the reversal here as well
            a_new[:, :, 1:i] += k.unsqueeze(2) * torch.flip(a[:, :, 1:i], dims=[2])

            a = a_new
            e = e * (1.0 - k**2)

        # Return LPC coefficients (excluding the leading 1)
        return a[:, :, 1:]

    def compute_group_delay(self, lpc_coeffs):
        """Compute group delay from LPC coefficients."""
        # lpc_coeffs: (batch_size, num_frames, order)
        batch_size, num_frames, order = lpc_coeffs.size()
        # Append 1 to the beginning of lpc_coeffs to make the denominator
        a = torch.cat(
            [
                torch.ones(batch_size, num_frames, 1, device=lpc_coeffs.device),
                lpc_coeffs,
            ],
            dim=2,
        )
        # Compute frequency response
        w = torch.linspace(0, np.pi, self.n_fft // 2 + 1, device=lpc_coeffs.device)
        w = w.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, n_freqs)
        ejw = torch.exp(
            -1j
            * w
            * torch.arange(order + 1, device=lpc_coeffs.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(3)
        )
        H = torch.sum(
            a.unsqueeze(3) * ejw, dim=2
        )  # Shape: (batch_size, num_frames, n_freqs)
        # Compute phase
        phase = torch.angle(H)
        # Unwrap phase
        # Compute group delay
        group_delay = -torch.diff(phase, dim=2)
        # Pad to match original length
        group_delay = F.pad(group_delay, (0, 1), mode="replicate")
        return group_delay.real  # Return the real part

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (batch_size, signal_length)
        # Move computations to the same device as input
        device = feat.device
        self.window = self.window.to(device)
        self.mel_filter_bank = self.mel_filter_bank.to(device)

        # Pre-emphasis
        feat = self.preemphasis(feat)

        # Frame the signal
        frames = self.frame_signal(
            feat, self.window_size, self.hop_length
        )  # (batch_size, num_frames, frame_length)

        # Apply window function
        frames = frames * self.window

        # Compute LPC coefficients
        order = 16
        lpc_coeffs = self.compute_lpc(frames, order)  # (batch_size, num_frames, order)

        # Compute group delay
        group_delays = self.compute_group_delay(
            lpc_coeffs
        )  # (batch_size, num_frames, n_freqs)

        # Apply Mel filterbank
        mel_group_delays = torch.matmul(
            group_delays, self.mel_filter_bank.T
        )  # (batch_size, num_frames, n_mels)

        # Take logarithm to compress dynamic range
        log_mel_group_delays = torch.log(mel_group_delays + 1e-8)

        output_lens = torch.full(
            (feat.size(0),),
            fill_value=log_mel_group_delays.size(1),
            dtype=torch.long,
            device=device,
        )

        return log_mel_group_delays, output_lens


class WaveletP(torch.nn.Module):
    def __init__(
        self,
        fs: int = 16000,
        n_filts: int = 80,
        n_fft: int = 512,
    ):
        super().__init__()
        self.fs = fs
        self.n_filts = n_filts
        self.n_fft = n_fft

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def extract_waveletp(wav):
            wp = pywt.WaveletPacket(
                data=wav, wavelet="db4", mode="symmetric", maxlevel=7
            )
            # Extract coefficients at the deepest
            nodes = wp.get_level(7, order="freq")
            coeffs = np.array([node.data for node in nodes])

            # Compute magnitude spectrum
            magnitude_spectrum = np.abs(coeffs) ** 2
            mel_filterbank = librosa.filters.mel(
                sr=16000, n_fft=254, n_mels=self.n_filts
            )
            mel_features = np.dot(mel_filterbank, magnitude_spectrum)
            return mel_features.T

        output = []
        output_lens = []
        for i, instance in enumerate(feat):
            waveletp = extract_waveletp(feat[i].cpu().numpy())
            output.append(torch.Tensor(waveletp))
            output_lens.append(waveletp.shape[0])
        # Hardcode again
        waveletp_feat = torch.stack(output, 0).cuda()
        output_lens = waveletp_feat.new_full(
            [waveletp_feat.size(0)], fill_value=waveletp_feat.size(1), dtype=torch.long
        )
        return waveletp_feat, output_lens.cuda()


class WaveletP_v2(torch.nn.Module):
    def __init__(
        self,
        fs: int = 16000,
    ):
        super().__init__()
        self.fs = fs

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def extract_waveletp(wav):
            wp = pywt.WaveletPacket(
                data=wav, wavelet="db4", mode="symmetric", maxlevel=7
            )
            # Extract coefficients at the deepest
            nodes = wp.get_level(7, order="freq")
            coeffs = np.array([node.data for node in nodes])

            return coeffs.T

        output = []
        output_lens = []
        for i, instance in enumerate(feat):
            waveletp = extract_waveletp(feat[i].cpu().numpy())
            output.append(torch.Tensor(waveletp))
            output_lens.append(waveletp.shape[0])
        # Hardcode again
        waveletp_feat = torch.stack(output, 0).cuda()
        output_lens = waveletp_feat.new_full(
            [waveletp_feat.size(0)], fill_value=waveletp_feat.size(1), dtype=torch.long
        )
        return waveletp_feat, output_lens.cuda()

#Symlet
class WaveletP_v3(torch.nn.Module):
    def __init__(
        self,
        fs: int = 16000,
    ):
        super().__init__()
        self.fs = fs

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def extract_waveletp(wav):
            wp = pywt.WaveletPacket(
                data=wav, wavelet="sym8", mode="symmetric", maxlevel=7
            )
            # Extract coefficients at the deepest
            nodes = wp.get_level(7, order="freq")
            coeffs = np.array([node.data for node in nodes])

            return coeffs.T

        output = []
        output_lens = []
        for i, instance in enumerate(feat):
            waveletp = extract_waveletp(feat[i].cpu().numpy())
            output.append(torch.Tensor(waveletp))
            output_lens.append(waveletp.shape[0])
        # Hardcode again
        waveletp_feat = torch.stack(output, 0).cuda()
        output_lens = waveletp_feat.new_full(
            [waveletp_feat.size(0)], fill_value=waveletp_feat.size(1), dtype=torch.long
        )
        return waveletp_feat, output_lens.cuda()


class WaveletP_v4(torch.nn.Module):
    def __init__(
        self,
        fs: int = 16000,
    ):
        super().__init__()
        self.fs = fs

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def extract_waveletp(wav):
            wp = pywt.WaveletPacket(
                data=wav, wavelet="coif5", mode="symmetric", maxlevel=7
            )
            # Extract coefficients at the deepest
            nodes = wp.get_level(7, order="freq")
            coeffs = np.array([node.data for node in nodes])

            return coeffs.T

        output = []
        output_lens = []
        for i, instance in enumerate(feat):
            waveletp = extract_waveletp(feat[i].cpu().numpy())
            output.append(torch.Tensor(waveletp))
            output_lens.append(waveletp.shape[0])
        # Hardcode again
        waveletp_feat = torch.stack(output, 0).cuda()
        output_lens = waveletp_feat.new_full(
            [waveletp_feat.size(0)], fill_value=waveletp_feat.size(1), dtype=torch.long
        )
        return waveletp_feat, output_lens.cuda()


class MODGD(torch.nn.Module):
    def __init__(
        self,
        fs: int = 16000,
        n_filts: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window_size: int = 512,
        preemph: float = 0.97,
        lifter: int = 6,
        gamma: float = 0.9,
        alpha: float = 0.4,
    ):
        super().__init__()
        self.fs = fs
        self.n_filts = n_filts
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.preemph = preemph
        self.lifter = lifter
        self.gamma = gamma
        self.alpha = alpha

        # using torchaudio.transforms
        self.mel_filterbanks = transforms.MelScale(
            sample_rate=self.fs,
            n_mels=self.n_filts,
            n_stft=self.n_fft // 2 + 1,
            mel_scale="slaney",
            norm="slaney",
        ).cuda()

    def preemphasis_torch(self, signal, coeff=0.97):
        """Apply pre-emphasis filter to the signal."""
        return torch.cat([signal[:1], signal[1:] - coeff * signal[:-1]])




    def framesig_torch(self,signal, winfunc=torch.hann_window):
        """Frame a signal into overlapping frames. Each time step t approximately at t * hop_length"""
        
        signal_dim = signal.dim()
        extended_shape = [1] * (3 - signal_dim) + list(signal.size())
        pad = int(self.n_fft // 2)
        signal = F.pad(signal.view(extended_shape), [pad, pad], "reflect") # reflection padding_mode + center 
        signal = signal.view(signal.shape[-signal_dim:])
        signal_length = len(signal)
        #
        num_frames = max(1, 1 + (signal_length - self.window_size) // self.hop_length)
        indices = torch.arange(0, self.window_size, device=signal.device).unsqueeze(0) + \
                torch.arange(0, num_frames * self.hop_length, self.hop_length, device=signal.device).unsqueeze(1)
        frames = signal[indices.long()]
        # Apply window function # torch.stft uses periodic hann by default
        window = winfunc(self.window_size, periodic=True).to(signal.device)

        return frames * window

    def get_complex_spec_torch(self, signal, with_time_scaled=True):
        """
        Compute the complex spectrum of the signal frames.
        """
        # Framing
        signal = self.preemphasis_torch(signal)
        frames = self.framesig_torch(signal)

        # FFT ; since the window_size = n_fft we don't need padding => centered frame
        complex_spec = torch.fft.rfft(frames, n=self.n_fft, dim=-1)

        time_scaled_complex_spec = None
        if with_time_scaled:
            time_indices = torch.arange(frames.size(1), device=frames.device)
            time_scaled_frames = frames * time_indices
            time_scaled_complex_spec = torch.fft.rfft(
                time_scaled_frames, n=self.n_fft, dim=-1
            )

        return complex_spec, time_scaled_complex_spec

    def cepstrally_smoothing_torch(self, spec):
        """
        Perform cepstral smoothing on the magnitude spectrum.
        """
        EPSILON = 1e-10  # Small constant to prevent log(0)
        # Handle zeros in spec
        spec = torch.maximum(spec, torch.tensor(EPSILON, device=spec.device))
        log_spec = torch.log(spec)

        # Inverse FFT to get cepstrum
        ceps = torch.fft.irfft(log_spec, n=self.n_fft, dim=-1)

        # Create liftering window
        lifter_window = (
            torch.arange(ceps.size(-1), device=ceps.device) < self.lifter
        ).float()
        lifter_window[self.lifter] = 0.5  # Edge smoothing at the lifter index

        # Expand lifter_window to match ceps dimensions
        lifter_window = lifter_window.unsqueeze(0).expand_as(ceps)

        # Apply liftering
        ceps_lifted = ceps * lifter_window

        # Reconstruct smoothed spectrum
        smoothed_spec = torch.abs(torch.fft.rfft(ceps_lifted, dim=-1))

        return smoothed_spec

    def dct_fft_impl(self, v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = self.dct_fft_impl(v)

        k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == "ortho":
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    def compute_modgd_features(self, signal):
        """
        Compute the MODGD features for a given signal.
        """
        # Compute complex spectra
        complex_spec, complex_spec_time_scaled = self.get_complex_spec_torch(signal)
        # Compute MODGD features
        features = self.get_modgdf(complex_spec, complex_spec_time_scaled)

        return features

    def get_modgdf(self, complex_spec, complex_spec_time_scaled):
        """
        Compute the Modified Group Delay Function (MODGD) features.
        """
        EPSILON = 1e-10
        # Magnitude spectrum
        mag_spec = torch.abs(complex_spec)
        mag_spec = torch.maximum(
            mag_spec, torch.tensor(EPSILON, device=mag_spec.device)
        )
        # Cepstral smoothing
        smoothed_mag_spec = self.cepstrally_smoothing_torch(mag_spec)
        # Real and imaginary parts
        real_spec = complex_spec.real
        imag_spec = complex_spec.imag
        real_spec_ts = complex_spec_time_scaled.real
        imag_spec_ts = complex_spec_time_scaled.imag

        # Numerator calculation
        numerator = (real_spec * real_spec_ts) + (imag_spec * imag_spec_ts)

        # Denominator calculation with smoothing
        denominator = smoothed_mag_spec ** (2.0 * self.gamma)

        # Compute the modified group delay function
        tao = numerator / denominator

        # Apply sign and magnitude scaling
        tao_sign = torch.sign(tao)
        tao_abs = torch.abs(tao)
        modgdf = tao_sign * (tao_abs**self.alpha)

        # Apply DCT to decorrelate features; or not
        #modgdf_features = self.dct(modgdf, norm="ortho")
        modgdf_features = modgdf.to(self.mel_filterbanks.fb.device)
        return self.mel_filterbanks(modgdf_features.T)

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the module."""

        def extract_modgd(wav):
            melmodgd = self.compute_modgd_features(wav)
            return melmodgd.T

        output = []
        output_lens = []
        for instance in feat:
            melmodgd = extract_modgd(
                instance.cuda()
            )  
            output.append(melmodgd)
            output_lens.append(melmodgd.shape[0])

        melmodgd_feat = torch.stack(output, 0).cuda()
        output_lens = melmodgd_feat.new_full(
            [melmodgd_feat.size(0)], fill_value=melmodgd_feat.size(1), dtype=torch.long
        )
        return melmodgd_feat, output_lens.cuda()


class Gamma_v2(torch.nn.Module):
    """Convert Raw audio to GAMMA"""

    def __init__(
        self,
        fs: int = 16000,
        n_filts: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        window_size: int = 512,
    ):
        super().__init__()
        self.fs = fs
        self.n_filts = n_filts
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.gammatone_filter_bank = torch.Tensor(gammatone_fbanks.gammatone_filter_banks(
            nfilts=self.n_filts, nfft=self.n_fft, fs=self.fs
        )[0]).cuda()
        self.amplitude_to_db_transform = transforms.AmplitudeToDB()

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: CQTS
        # (B,T) -> (B,D,n_mels)
        def extract_gamma(wav):
            magnitude = torch.abs(torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_size,
                        center=True, pad_mode="reflect", return_complex=True,window=torch.windows.hann(self.window_size).to(wav)))**2

            Gam = torch.matmul(self.gammatone_filter_bank.cuda(), magnitude) # During the stats collect, the module is manually transferred to "cpu" device

            return self.amplitude_to_db_transform(Gam).T

        output = []
        output_lens = []
        for instance in feat:
            gamma = extract_gamma(instance.cuda())
            output.append(gamma)
            output_lens.append(gamma.shape[0])
        # Hardcode again
        gamma_feat = torch.stack(output, 0).cuda()
        output_lens = gamma_feat.new_full(
            [gamma_feat.size(0)], fill_value=gamma_feat.size(1), dtype=torch.long
        )
        return gamma_feat, output_lens.cuda()

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class LogMel_default(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        melmat = librosa.filters.mel(**_mel_options)
        # melmat: (D2, D1) -> (D1, D2)
        self.register_buffer("melmat", torch.from_numpy(melmat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        mel_feat = torch.matmul(feat, self.melmat)
        mel_feat = torch.clamp(mel_feat, min=1e-10)

        if self.log_base is None:
            logmel_feat = mel_feat.log()
        elif self.log_base == 2.0:
            logmel_feat = mel_feat.log2()
        elif self.log_base == 10.0:
            logmel_feat = mel_feat.log10()
        else:
            logmel_feat = mel_feat.log() / torch.log(self.log_base)

        # Zero padding
        if ilens is not None:
            logmel_feat = logmel_feat.masked_fill(
                make_pad_mask(ilens, logmel_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmel_feat, ilens
