#!/usr/bin/env bash

. ./path.sh
. ./cmd.sh
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
inference_nj=8
inference_bin_tag=ensemble_trial
inference_tag=decoding_ensemble
path_to_conf=e_conf/testing_conf.yaml
bpemode=unigram
local_score_opts= 
token_type=bpe
cleaner=null    
hyp_cleaner=none
nbpe=5000
valid_set="dev"
test_sets="test_other"
eval_valid_set=false
dumpdir=dump
data_feats=${dumpdir}/raw
asr_exp=ensemble_exp
python=python3
lang=en
nlsyms_txt=none
num_ref=1
num_inf=
num_inf=${num_inf:=${num_ref}}
_ngpu=0
_cmd=${decode_cmd}
if [ ${num_ref} -eq 1 ]; then
    # For single speaker, text file path and name are text
    ref_text_files_str="text "
    ref_text_names_str="text "
else
    # For multiple speakers, text file path and name are text_spk[1-N] and [text, text_spk2, ...]
    #TODO(simpleoier): later to support flexibly defined text prefix
    ref_text_files_str="text_spk1 "
    ref_text_names_str="text "
    for n in $(seq 2 ${num_ref}); do
        ref_text_files_str+="text_spk${n} "
        ref_text_names_str+="text_spk${n} "
    done
fi
# shellcheck disable=SC2206
ref_text_files=(${ref_text_files_str// / })

. utils/parse_options.sh


if "${eval_valid_set}"; then
    _dsets="org/${valid_set} ${test_sets}"
else
    _dsets="${test_sets}"
fi
for dset in ${_dsets}; do
    _data="${data_feats}/${dset}"
    _dir="${asr_exp}/${inference_tag}/${dset}"
    _logdir="${_dir}/logdir"
    mkdir -p "${_logdir}"

    _feats_type="$(<${_data}/feats_type)"
    _audio_format="$(cat ${_data}/audio_format 2>/dev/null || echo ${audio_format})"
    if [ "${_feats_type}" = raw ]; then
        _scp=wav.scp
        if [[ "${audio_format}" == *ark* ]]; then
            _type=kaldi_ark
        elif [[ "${_audio_format}" == *multi* ]]; then
            _type=multi_columns_sound
        else
            _type=sound
        fi
    else
        _scp=feats.scp
        _type=kaldi_ark
    fi


    key_file=${_data}/${_scp}

    split_scps=""

    _nj="${inference_nj}" 

    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    utils/split_scp.pl "${key_file}" ${split_scps}

    rm -f "${_logdir}/*.log"
    # shellcheck disable=SC2046,SC2086
    ${_cmd}  JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
        python3 evaluate_late_fusion.py \
        --path "${path_to_conf}" \
        --data_path_and_name_and_type "${_data}/${_scp}, speech, kaldi_ark" \
        --key_file "${_logdir}"/keys.JOB.scp \
        --output_dir "${_logdir}"/output.JOB
done

for ref_txt in ${ref_text_files[@]}; do
    suffix=$(echo ${ref_txt} | sed 's/text//')
    for f in token token_int score text; do
        if [ -f "${_logdir}/output.1/1best_recog/${f}${suffix}" ]; then
            for i in $(seq "${_nj}"); do
                cat "${_logdir}/output.${i}/1best_recog/${f}${suffix}"
            done | sort -k1 >"${_dir}/${f}${suffix}"
        fi
    done
done
 # Define inference tag for each trial

#If I assume starting folder is ensemble_exp/

# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi
bpedir="${token_listdir}/bpe_${bpemode}${nbpe}"
bpeprefix="${bpedir}"/bpe
bpemodel="${bpeprefix}".model
score_opts= 


if [ "${token_type}" = phn ]; then
    log "Error: Not implemented for token_type=phn"
    exit 1
fi

if "${eval_valid_set}"; then
    _dsets="org/${valid_set} ${test_sets}"
else
    _dsets="${test_sets}"
fi
for dset in ${_dsets}; do
    _data="${data_feats}/${dset}"
    _dir="${asr_exp}/${inference_tag}/${dset}"

    for _tok_type in "char" "word" "bpe"; do
        [ "${_tok_type}" = bpe ] && [ ! -f "${bpemodel}" ] && continue

        _opts="--token_type ${_tok_type} "
        if [ "${_tok_type}" = "char" ] || [ "${_tok_type}" = "word" ]; then
            _type="${_tok_type:0:1}er"
            _opts+="--non_linguistic_symbols ${nlsyms_txt} "

            _opts+="--remove_non_linguistic_symbols true "

        elif [ "${_tok_type}" = "bpe" ]; then
            _type="ter"
            _opts+="--bpemodel ${bpemodel} "

        else
            log "Error: unsupported token type ${_tok_type}"
        fi

        _scoredir="${_dir}/score_${_type}"
        mkdir -p "${_scoredir}"

        # shellcheck disable=SC2068
        for ref_txt in ${ref_text_files[@]}; do
            # Note(simpleoier): to get the suffix after text, e.g. "text_spk1" -> "_spk1"
            suffix=$(echo ${ref_txt} | sed 's/text//')
            # Tokenize text to ${_tok_type} level
            paste \
                <(<"${_data}/${ref_txt}" \
                    ${python} -m espnet2.bin.tokenize_text  \
                        -f 2- --input - --output - \
                        --cleaner "${cleaner}" \
                        ${_opts} \
                        ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/ref${suffix:-${suffix}}.trn"

            # NOTE(kamo): Don't use cleaner for hyp
            paste \
                <(<"${_dir}/${ref_txt}"  \
                    ${python} -m espnet2.bin.tokenize_text  \
                        -f 2- --input - --output - \
                        ${_opts} \
                        --cleaner "${hyp_cleaner}" \
                        ) \
                <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                    >"${_scoredir}/hyp${suffix:-${suffix}}.trn"

        done

        # Note(simpleoier): score across all possible permutations
        if [ ${num_ref} -gt 1 ] && [ -n "${suffix}" ]; then
            for i in $(seq ${num_ref}); do
                for j in $(seq ${num_inf}); do
                    sclite \
                        ${score_opts} \
                        -r "${_scoredir}/ref_spk${i}.trn" trn \
                        -h "${_scoredir}/hyp_spk${j}.trn" trn \
                        -i rm -o all stdout > "${_scoredir}/result_r${i}h${j}.txt"
                done
            done
            # Generate the oracle permutation hyp.trn and ref.trn
            pyscripts/utils/eval_perm_free_error.py --num-spkrs ${num_ref} \
                --results-dir ${_scoredir}
        fi

        sclite \
            ${score_opts} \
            -r "${_scoredir}/ref.trn" trn \
            -h "${_scoredir}/hyp.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result.txt"

        grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
    done
done

[ -f local/score.sh ] && local/score.sh ${local_score_opts} "${asr_exp}"

# Show results in Markdown syntax
scripts/utils/show_asr_result.sh "${asr_exp}" > "${asr_exp}"/RESULTS.md
cat "${asr_exp}"/RESULTS.md
