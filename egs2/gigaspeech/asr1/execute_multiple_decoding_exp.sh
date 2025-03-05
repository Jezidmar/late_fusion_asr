#!/usr/bin/env bash 

#./execute_decoding.sh --inference_nj 4  --inference_tag mel_gamma_beamsize60_0.75_0.65 --path_to_conf e_conf/mel_gamma.yaml
#./execute_decoding.sh --inference_nj 3 --inference_tag mel_cqt_bark_late_fusion --path_to_conf e_conf/mel_cqt_bark.yaml
#./execute_decoding.sh --inference_nj 2 --inference_tag mel_cqt_mfcc_late_fusion_beamsize10_oldapproach --path_to_conf e_conf/mel_cqt_mfcc.yaml
#./execute_decoding.sh --inference_nj 3 --inference_tag mel_mfcc_gamma_cqt_lm_2 --path_to_conf e_conf/mel_mfcc_gamma_cqt_lm.yaml
#./execute_decoding.sh --inference_nj 3 --inference_tag mel_modgd  --path_to_conf e_conf/mel_modgd.yaml
#./execute_decoding.sh --inference_nj 3 --inference_tag mel_modgd_lm  --path_to_conf e_conf/mel_modgd_lm.yaml
#./execute_decoding.sh --inference_nj 2 --inference_tag mel_mfcc_gamma_daubechies --path_to_conf e_conf/mel_mfcc_gamma_daubechies.yaml
#./execute_decoding.sh --inference_nj 2 --inference_tag mel_mfcc_gamma_simlet --path_to_conf e_conf/mel_mfcc_gamma_simlet.yaml
#./execute_decoding.sh --inference_nj 2 --inference_tag mel_mfcc_gamma_cqt_beam7 --path_to_conf e_conf/mel_mfcc_gamma_cqt_beam7.yaml
#./execute_decoding.sh --inference_nj 2 --inference_tag mel_mfcc_gamma_cqt_beam3 --path_to_conf e_conf/mel_mfcc_gamma_cqt_beam3.yaml
#./execute_decoding.sh --inference_nj 2 --inference_tag mel_modgd_gamma_cqtDefMelLM --path_to_conf e_conf/mel_modgd_gamma_cqt_lm.yaml
#./execute_decoding.sh --inference_nj 3 --inference_tag mel_modgd_gamma_mfccCalibratedv1 --path_to_conf e_conf/mel_modgd_gamma_mfcc.yaml
./execute_decoding.sh --inference_nj 3 --inference_tag mel_modgd_gamma_cqtDefMel --path_to_conf e_conf/mel_modgd_gamma_cqt.yaml --test_sets dev
#./execute_decoding.sh --inference_nj 4 --inference_tag mel_modgdDefMel --path_to_conf e_conf/mel_modgd.yaml




#./execute_decoding.sh --inference_nj 3 --inference_tag mel_mfcc_gamma --path_to_conf e_conf/mel_mfcc_gamma.yaml
#./execute_decoding.sh --inference_nj 6 --inference_tag mel_gamma --path_to_conf e_conf/mel_gamma.yaml

#./execute_decoding.sh --inference_nj 2 --inference_tag mel_gamma_beamsize10_0.75_0.65_old_approach --path_to_conf e_conf/mel_gamma.yaml
#./execute_decoding.sh --inference_nj 6  --inference_tag mel_simlet --path_to_conf e_conf/mel_simlet.yaml
