### This is the code related to the publication: Late fusion ensembles for speech recognition on diverse input audio representations


 Paper has been accepted to special session on Data Science: Foundations and Applications (DSFA) of conference PAKDD 2025.

## Experiments

- Experiments are conducted on 4 datasets
   - Librispeech
   - Aishell1
   - Tedlium_v2
   - GigaSpeech


### Download tokens dictionary for coherent behaviour
```bash
  cd ../espnet/egs2/librispeech/asr1/data/en_token_list/
  gdown --folder --id 1rrr_pymgRQ33YGHdimu15tYEsv0-NnvO?usp=sharing
  ```

### Training basic model with specific features
- Enter `/egs2/librispeech/asr1/conf/tuning` folder and find demo train_asr_e_branchformer.yaml. Within `frontend_conf` switch feat type according to needs. Complete list of features can be found in `/espnet2/layers/log_mel.py`


