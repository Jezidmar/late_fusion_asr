#!/usr/bin/env bash 

./execute_decoding.sh --inference_nj 3 --inference_tag mel_gamma_bark_late_fusion --path_to_conf e_conf/mel_gamma_bark.yaml
./execute_decoding.sh --inference_nj 3 --inference_tag mel_cqt_bark_late_fusion --path_to_conf e_conf/mel_cqt_bark.yaml
./execute_decoding.sh --inference_nj 3 --inference_tag mel_cqt_mfcc_late_fusion --path_to_conf e_conf/mel_cqt_mfcc.yaml
./execute_decoding.sh --inference_nj 3 --inference_tag mel_cqt_gamma_late_fusion --path_to_conf e_conf/mel_cqt_gamma.yaml
./execute_decoding.sh --inference_nj 3 --inference_tag mel_gamma_mfcc_late_fusion --path_to_conf e_conf/mel_gamma_mfcc.yaml
