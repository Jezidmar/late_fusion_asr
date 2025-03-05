import argparse
import gc
from typing import List, Optional

import torch
import yaml
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.log_mel import Bark, Cqt, Gamma, LogMel, Mfcc
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.utils.types import str2triple_str, str_or_none
from modified_beam_search import EnsembleBeamSearch
from pydantic import BaseModel

from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus


class WeightsDict(BaseModel):
    lm: Optional[float] = None
    decoder_mel: Optional[float] = None
    decoder_cqt: Optional[float] = None
    decoder_mfcc: Optional[float] = None
    decoder_gamma: Optional[float] = None
    decoder_bark: Optional[float] = None
    ctc_mel: Optional[float] = None
    ctc_cqt: Optional[float] = None
    ctc_mfcc: Optional[float] = None
    ctc_gamma: Optional[float] = None
    ctc_bark: Optional[float] = None
    ngram: Optional[float] = None
    length_bonus: Optional[float] = None


class EnsembleEvaluationConfig(BaseModel):
    weights: dict
    beam_size: int = 5
    device: str = "cpu"
    num_workers: int = 12
    testing_sets: List[str] = ["test_clean", "test_other"]


class DecodeEnseble:
    def __init__(self, config_path):
        with open(config_path, "r") as stream:
            self.config = yaml.safe_load(
                stream
            )  # config is the .yaml file comprising weights for specific feature for the decoding process

    def _preprocess_loaded_audio(self, wave, extractors, global_mvn_stats_files):
        "Shape of [1, T] is accepted here as wave file"
        features = {}
        for key, ext in extractors.items():
            features[key] = ext(wave, None)[0]
            intermediate_out1 = global_mvn_stats_files[key](features[key])
            features[key] = (
                features[key][0].unsqueeze(0),
                torch.tensor([int(elem) for elem in intermediate_out1[1]]),
            )
        return features

    def _recognize(
        self,
        model_dict,
        asr_train_args,
        beam_search,
        wave,
        device,
        extractors,
        global_mvn_stats_files,
    ):
        feats_dict = self._preprocess_loaded_audio(
            wave, extractors, global_mvn_stats_files
        )

        encoder_outs = {}
        for key, model in model_dict.items():
            key_dec = f"decoder_{key}"
            encoder_out, encoder_out_lens, _ = model.encoder(
                feats_dict[key][0].to(device), feats_dict[key][1].to(device)
            )
            encoder_outs[key_dec] = encoder_out[0].to(device)

        hyp = beam_search(x=encoder_outs, maxlenratio=0.6, minlenratio=0.0)
        converter = TokenIDConverter(token_list=asr_train_args.token_list)
        tokenizer = build_tokenizer(
            token_type=asr_train_args.token_type, bpemodel=asr_train_args.bpemodel
        )

        results = []
        last_pos = -1
        if isinstance(hyp.yseq, list):
            token_int = hyp.yseq[1:last_pos]
        else:
            token_int = hyp.yseq[1:last_pos].tolist()

        # remove blank symbol id, which is assumed to be 0
        token_int = list(filter(lambda x: x != 0, token_int))

        token = converter.ids2tokens(token_int)
        if tokenizer is not None:
            text = tokenizer.tokens2text(token)
        else:
            text = None
        # print(text)
        hyp = hyp.detach()
        results.append((text, token, token_int, hyp))
        del hyp
        gc.collect()
        return results

    def decode(self):
        weights_conf = WeightsDict(**self.config["weights"])
        conf = EnsembleEvaluationConfig(**self.config)
        model_dict = {}
        scorers = {}
        extractors = {}
        global_mvn_stats_files = {}
        if getattr(weights_conf, "lm") is not None:
            from espnet2.tasks.lm import LMTask

            lm_train_config_path = "config_files/lm_config.yaml"
            lm_file = "models/4epoch.pth"
            lm, lm_train_args = LMTask.build_model_from_file(
                lm_train_config_path, lm_file, conf.device
            )
            lm.eval()
            scorers["lm"] = lm.lm
        if getattr(weights_conf, "decoder_mel") is not None:
            model_path = "models/mel_valid.acc.ave.pth"
            train_config = "config_files/mel_config.yaml"
            model, asr_train_args = ASRTask.build_model_from_file(
                train_config, model_path, conf.device
            )
            model.eval()
            model_dict["mel"] = model
            scorers["decoder_mel"] = model.decoder
            scorers["ctc_mel"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
            extractors["mel"] = LogMel(
                fs=16000,
                n_fft=512,
                n_mels=80,
                hop_length=160,
            )
            global_mvn_stats_files["mel"] = GlobalMVN(stats_file="stats/mel_stats.npz")
        if getattr(weights_conf, "decoder_mfcc") is not None:
            model_path = "models/mfcc_valid.acc.ave.pth"
            train_config = "config_files/mfcc_config.yaml"
            model, asr_train_args = ASRTask.build_model_from_file(
                train_config, model_path, conf.device
            )
            model.eval()
            model_dict["mfcc"] = model
            scorers["decoder_mfcc"] = model.decoder
            scorers["ctc_mfcc"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
            extractors["mfcc"] = Mfcc(fs=16000, n_fft=512, n_mels=80, hop_length=160)
            global_mvn_stats_files["mfcc"] = GlobalMVN(
                stats_file="stats/mfcc_stats.npz"
            )
        if getattr(weights_conf, "decoder_cqt") is not None:
            model_path = "models/cqt_valid.acc.ave.pth"
            train_config = "config_files/cqt_config.yaml"
            model, asr_train_args = ASRTask.build_model_from_file(
                train_config, model_path, conf.device
            )
            model.eval()
            model_dict["cqt"] = model
            scorers["decoder_cqt"] = model.decoder
            scorers["ctc_cqt"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
            extractors["cqt"] = Cqt(
                fs=16000,
                n_bins=80,
                hop_length=160,
            )
            global_mvn_stats_files["cqt"] = GlobalMVN(stats_file="stats/cqt_stats.npz")
        if getattr(weights_conf, "decoder_gamma") is not None:
            model_path = "models/gamma_valid.acc.ave.pth"
            train_config = "config_files/gamma_config.yaml"
            model, asr_train_args = ASRTask.build_model_from_file(
                train_config, model_path, conf.device
            )
            model.eval()
            model_dict["gamma"] = model
            scorers["decoder_gamma"] = model.decoder
            scorers["ctc_gamma"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
            extractors["gamma"] = Gamma(
                fs=16000,
                n_fft=512,
                n_filts=80,
                window_size=400,
                hop_length=160,
            )
            global_mvn_stats_files["gamma"] = GlobalMVN(
                stats_file="stats/gamma_stats.npz"
            )
        if getattr(weights_conf, "decoder_bark") is not None:
            model_path = "models/gamma_valid.acc.ave.pth"
            train_config = "config_files/bark_config.yaml"
            model, asr_train_args = ASRTask.build_model_from_file(
                train_config, model_path, conf.device
            )
            model.eval()
            model_dict["bark"] = model
            scorers["decoder_bark"] = model.decoder
            scorers["ctc_bark"] = CTCPrefixScorer(ctc=model.ctc, eos=model.eos)
            extractors["bark"] = Bark(
                fs=16000,
                n_fft=512,
                n_filts=80,
                hop_length=160,
                window_size=400,
            )
            global_mvn_stats_files["bark"] = GlobalMVN(
                stats_file="stats/bark_stats.npz"
            )
        nbest = 1
        scorers["length_bonus"] = LengthBonus(len(model.token_list))

        weights = self.config["weights"]

        dtype = "float32"

        data_path_and_name_and_type = args.data_path_and_name_and_type
        # print(data_path_and_name_and_type)
        loader = ASRTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=1,
            key_file=args.key_file,
            num_workers=conf.num_workers,
            preprocess_fn=ASRTask.build_preprocess_fn(asr_train_args, False),
            collate_fn=ASRTask.build_collate_fn(asr_train_args, False),
            allow_variable_data_keys=True,
            inference=True,
        )
        output_dir = args.output_dir

        with DatadirWriter(output_dir) as writer:
            for keys, batch in loader:
                assert isinstance(batch, dict), type(batch)
                assert all(isinstance(s, str) for s in keys), keys
                _bs = len(next(iter(batch.values())))
                assert len(keys) == _bs, f"{len(keys)} != {_bs}"

                beam_search = EnsembleBeamSearch(
                    beam_size=conf.beam_size,
                    weights=weights,
                    scorers=scorers,
                    sos=model.sos,
                    eos=model.eos,
                    vocab_size=len(model.token_list),
                    token_list=model.token_list,
                    pre_beam_score_key="full",
                    normalize_length=False,
                )

                batch = {
                    k: v[0] for k, v in batch.items() if not k.endswith("_lengths")
                }
                wave = batch["speech"].unsqueeze(0)

                with torch.no_grad():
                    results = self._recognize(
                        model_dict,
                        asr_train_args,
                        beam_search,
                        wave,
                        conf.device,
                        extractors,
                        global_mvn_stats_files,
                    )
                if conf.device == "cuda":
                    torch.cuda.empty_cache()
                else:
                    gc.collect()
                    del batch
                key = keys[0]

                # Normal ASR
                encoder_interctc_res = None
                if isinstance(results, tuple):
                    results, encoder_interctc_res = results

                for n, (text, token, token_int, hyp) in zip(
                    range(1, nbest + 1), results
                ):
                    # Create a directory: outdir/{n}best_recog
                    ibest_writer = writer[f"{n}best_recog"]
                    # ibest_writer = writer # Don't create this directory
                    # Write the result to each file
                    ibest_writer["token"][key] = " ".join(token)
                    ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                    ibest_writer["score"][key] = str(hyp.score)

                    if text is not None:
                        ibest_writer["text"][key] = text

                # Write intermediate predictions to
                # encoder_interctc_layer<layer_idx>.txt
                ibest_writer = writer["1best_recog"]
                # ibest_writer = writer
                if encoder_interctc_res is not None:
                    for idx, text in encoder_interctc_res.items():
                        ibest_writer[f"encoder_interctc_layer{idx}.txt"][key] = (
                            " ".join(text)
                        )
                del results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to decoding config")
    parser.add_argument(
        "--data_path_and_name_and_type", type=str2triple_str, action="append"
    )
    parser.add_argument("--key_file", type=str_or_none)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    func = DecodeEnseble(args.path)
    func.decode()  # We assume the config file is in same directory
