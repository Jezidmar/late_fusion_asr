import copy
from typing import Optional, Tuple, Union,List

import humanfriendly
import numpy as np
import torch
from typeguard import check_argument_types
from espnet2.layers.stft import Stft
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.log_mel import Mfcc
from espnet2.layers.log_mel import Cqt
from espnet2.layers.log_mel import Gamma, Bark, GroupDelay,WaveletP,WaveletP_v2,WaveletP_v3,WaveletP_v4,MODGD
from torch_complex.tensor import ComplexTensor
from typeguard import typechecked

from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class DefaultFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    # Hardcoded feature extractor for MEL,MFCC,CQT,GAMMA features.
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        window_size: int = 400,
        hop_length: int = 160,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        feat_type: str = "mel",
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length
        self.window_size = window_size
        self.n_mels = n_mels
        self.frontend_type = "default"

        if feat_type == "mel":
            self.extract_feats = (
                LogMel(  # <--Just replace LogMel with Gamma, Mfcc, Cqt, GroupD,LogMel
                    fs=fs,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    hop_length=self.hop_length,
                )
            )
        elif feat_type == "cqt":
            self.extract_feats = Cqt(
                fs=fs,
                n_bins=n_mels,
                hop_length=self.hop_length,
            )
        elif feat_type == "mfcc":
            self.extract_feats = Mfcc(
                fs=fs,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=self.hop_length,
            )
        elif feat_type == "gamma":
            self.extract_feats = Gamma(
                fs=fs,
                n_fft=n_fft,
                n_filts=n_mels,
                window_size=self.window_size,
                hop_length=self.hop_length,
            )

        elif feat_type == "bark":
            self.extract_feats = Bark(
                fs=fs,
                n_fft=n_fft,
                n_filts=n_mels,
                hop_length=self.hop_length,
                window_size= self.window_size,
            )
        elif feat_type == "phase":
            self.extract_feats = GroupDelay(
                fs=fs,
                n_fft=n_fft,
                n_filts=n_mels,
                hop_length=self.hop_length,
                window_size= self.window_size,
            )
        elif feat_type == "wt":
            self.extract_feats = WaveletP(
                fs=fs,
                n_fft=n_fft,
                n_filts=n_mels,
            )
        elif feat_type == "wt2":
            self.extract_feats = WaveletP_v2(
                fs=fs,
            )
        elif feat_type == "wt3":
            self.extract_feats = WaveletP_v3(
                fs=fs,
            )
        elif feat_type == "wt4":
            self.extract_feats = WaveletP_v4(
                fs=fs,
            )
        elif feat_type == "modgd":
            self.extract_feats = MODGD(
                fs = fs,
            )

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        # Input lengths are ;None; <-Hardcoded
        # Everything is hardcoded for now. Input is:
        input_feats, feats_lens = self.extract_feats(input)
        #return input_feats.cpu(), feats_lens.cpu()  #<--for inferencing
        return input_feats.cuda(), feats_lens.cuda()  # <--for training


class DefaultFrontendDEF(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Log-Mel-Fbank
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        window_size: Optional[int] = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: Optional[int] = None,
        fmax: Optional[int] = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
    ):
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                window_size=window_size,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        return input_feats, feats_lens

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens


class FeatFusedFrontend(AbsFrontend):
    """Fused frontend structure using multiple feature extraction methods.

    This frontend extracts features using LogMel, MFCC, CQT, Gamma, and GroupD.
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        window_size: int = 400,
        hop_length: int = 160,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        extractors: Optional[List[str]] = None
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.extractors=extractors
        self.window_size=window_size
        funcs={}

        # Initialize the different feature extractors
        if "mel" in self.extractors:
            self.logmel_extractor = LogMel(  # <--Just replace LogMel with Gamma, Mfcc, Cqt, GroupD,LogMel
                    fs=fs,
                    n_fft=n_fft,
                    n_mels=n_mels,
                    hop_length=self.hop_length,
                )
            funcs["mel"]=self.logmel_extractor
        if "mfcc" in self.extractors:
            self.mfcc_extractor = Mfcc(
                fs=fs,
                n_fft=n_fft,
                n_mels=n_mels,
                hop_length=self.hop_length,
            )
            funcs["mfcc"]=self.mfcc_extractor
        if "cqt" in self.extractors:
            self.cqt_extractor = Cqt(
                fs=fs,
                n_bins=n_mels,
                hop_length=self.hop_length,
            )
            funcs["cqt"]=self.cqt_extractor
        if "gamma" in self.extractors:
            self.gamma_extractor = Gamma(
                fs=fs,
                n_fft=n_fft,
                n_filts=n_mels,
                window_size=self.window_size,
                hop_length=self.hop_length,
            )
            funcs["gamma"]=self.gamma_extractor
        if "bark" in self.extractors:
            self.bark_extractor = Bark(
                fs=fs,
                n_fft=n_fft,
                n_filts=n_mels,
                hop_length=self.hop_length,
                window_size= self.window_size,
            )
            funcs["bark"]=self.bark_extractor

        if "daubechies" in self.extractors:
            self.daubechies_extractor = WaveletP_v2(
                fs = fs
            )
            funcs["daubechies"]=self.bark_extractor
        self.functions = funcs
    def output_size(self) -> int:
        return self.n_mels

    def forward(self, input: torch.Tensor, input_lengths: torch.Tensor):
        # Extract features using all methods
        out = []
        for key,func in self.functions.items():
            feats,feat_lens = func(input)
            out.append(feats)

        out_feats = [feat.cuda() for feat in out]
        del feats, input, out

        return out_feats, feat_lens
