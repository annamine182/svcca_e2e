#!/bin/bash

set -e -o pipefail


#EXPERIMENT MODIFIERS
stage=4
free_gpu=$CUDA_VISIBLE_DEVICES
gpu_split_array=(${CUDA_VISIBLE_DEVICES//,/ })
ngpus=${#gpu_split_array[@]}
echo "ngpus $ngpus"

name=lstm_conv_6_layers
#experiment=/share/mini1/sw/spl/espresso/git20201130/examples/asr_swbd/exp/transformer_spec_aug/*
#experiment=/share/mini1/sw/spl/espresso/git20201130/examples/asr_swbd/exp/transformer_transformer_2021/*
#experiment=/share/mini1/sw/spl/espresso/git20210123/examples/asr_swbd/exp/lstm_conv_x0.5/*
#experiment=/share/mini1/sw/spl/espresso/git20210123/examples/asr_swbd/exp/lstm_conv_x1.5/*
#experiment=/share/mini1/sw/spl/espresso/git20210123/examples/asr_swbd/exp/lstm_conv_x0.25/*
#experiment=/share/mini1/sw/spl/espresso/git20210123/examples/asr_swbd/exp/$name/*
#experiment=/share/mini1/sw/spl/espresso/git20210123/examples/asr_swbd/exp/lstm_conv_5_layers_x0.5/*
#experiment=/share/mini1/sw/spl/espresso/git20210123/examples/asr_swbd/exp/lstm_conv_3_layers/*
#experiment=/share/mini1/sw/spl/espresso/git20210123/examples/asr_swbd/exp/lstm_conv_3_layers_x1.5/*
experiment=/share/mini1/sw/spl/espresso/multi-stream/mchan/examples/asr_swbd/exp/transformer_13_encoder/*

# E2E model related
affix=
train_set=train
valid_set=
test_set="eval2000"
use_transformer=true

# LM related
lm_affix=
lm_checkpoint=checkpoint_best.pt
lm_shallow_fusion=false # no LM fusion if false
sentencepiece_vocabsize=1000
sentencepiece_type=unigram

# data related
dumpdir=data/dump   # directory to dump full features
swbd1_dir=/share/mini1/sw/spl/espresso/new_svcca/svcca/data/sub_train
eval2000_dir="/share/mini1/sw/spl/espresso/new_svcca/svcca/data/sub_eval2000 /share/mini1/data/audvis/pub/asr/cts/us/eval00/LDC2002T43"


# feature configuration
do_delta=false
apply_specaug=false

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

lmdir=exp/lm_lstm${lm_affix:+_${lm_affix}}
dir=exp/lstm${affix:+_$affix}

# Note stage0 for eval2000 not complete and need to link text directory
if [ $stage -le 0 ]; then
 echo "Running data download"
 /share/mini1/sw/spl/espresso/new_svcca/svcca/local/swbd1_data_download.sh ${swbd1_dir}
 echo "Running dict prep"
 /share/mini1/sw/spl/espresso/new_svcca/svcca/local/swbd1_prepare_dict.sh
 echo "Running swbd1 data prep"
 /share/mini1/sw/spl/espresso/new_svcca/svcca/local/swbd1_data_prep.sh ${swbd1_dir}
 echo "Running eval2000 data prep"
 /share/mini1/sw/spl/espresso/new_svcca/svcca/local/eval2000_data_prep.sh ${eval2000_dir}
 echo "Eval2000 data prep script ok"
 for x in eval2000; do
   cp data/${x}/text data/${x}/text.org
   paste -d "" \
     <(cut -f 1 -d" " data/${x}/text.org) \
     <(awk '{$1=""; print tolower($0)}' data/${x}/text.org | perl -pe 's| \(\%.*\)||g' | perl -pe 's| \<.*\>||g' | sed -e "s/(//g" -e "s/)//g") \
     | sed -e 's/\s\+/ /g' > data/${x}/text
 done
 echo "Succeeded in formatting data."
 echo "Stage 0 Completed"
fi

train_feat_dir=$dumpdir/$train_set/delta${do_delta}; mkdir -p $train_feat_dir



if [ $stage -le 1 ]; then
 echo "Stage 1: Feature generation"
 fbankdir=fbank
 for x in train eval2000; do
   steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write-utt2num-frames true \
   data/$x exp/make_fbank/$x $fbankdir
   utils/fix_data_dir.sh data/$x
 done

 echo "done fbank"
 # Compute global CMVN
 compute-cmvn-stats scp:data/$train_set/feats.scp data/$train_set/cmvn.ark
 echo "computed cmvn"
 # dump features
 dump.sh --cmd "$train_cmd" --nj 32 --do_delta $do_delta \
   data/$train_set/feats.scp data/$train_set/cmvn.ark exp/dump_feats/train $train_feat_dir
 for rtask in $test_set; do
   test_feat_dir=$dumpdir/$rtask/delta${do_delta}; mkdir $test_feat_dir 
   dump.sh --cmd "$train_cmd" --nj 10 --do_delta $do_delta \
     data/$rtask/feats.scp data/$train_set/cmvn.ark exp/dump_feats/recog/$rtask \
     $test_feat_dir
 done
 echo "dumped features"
fi



dict=data/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}_units.txt
sentencepiece_model=data/lang/${train_set}_${sentencepiece_type}${sentencepiece_vocabsize}
nlsyms=data/lang/non_lang_syms.txt
lmdatadir=data/lm_text
lmdict=$dict

if [ $stage -le 2 ]; then
  echo "Stage 2: Text tokenisation"
  echo "$0: making a non-linguistic symbol list..."
  train_text=data/$train_set/text
  cut -f 2- $train_text | tr " " "\n" | sort | uniq | grep "\[" > $nlsyms
  cat $nlsyms
 
 echo "$0: tokenizing text for train/valid/test sets..."
  for dataset in $train_set $test_set; do  # validation is included in tests
    text=data/$dataset/text
    token_text=data/$dataset/token_text
    cut -f 2- -d" " $text | \
      python3 ../scripts/spm_encode.py --model=${sentencepiece_model}.model --output_format=piece | \
      paste -d" " <(cut -f 1 -d" " $text) - > $token_text
    cut -f 2- -d" " $token_text > $lmdatadir/$dataset.tokens
  done
fi

if [ $stage -le 3 ]; then
  echo "Stage 3: Dumping Json Files"
  train_feat=$train_feat_dir/feats.scp
  train_token_text=data/$train_set/token_text
  train_utt2num_frames=data/$train_set/utt2num_frames
  valid_feat=$valid_feat_dir/feats.scp
  valid_token_text=data/$valid_set/token_text
  valid_utt2num_frames=data/$valid_set/utt2num_frames
  asr_prep_json.py --feat-files $train_feat --token-text-files $train_token_text --utt2num-frames-files $train_utt2num_frames --output data/train.json
  for dataset in $test_set; do
    feat=${dumpdir}/$dataset/delta${do_delta}/feats.scp
    utt2num_frames=data/$dataset/utt2num_frames
    # only score train_dev with built-in scorer
    text_opt= && [ "$dataset" == "train_dev" ] && text_opt="--token-text-files data/$dataset/token_text"
    asr_prep_json.py --feat-files $feat $text_opt --utt2num-frames-files $utt2num_frames --output data/$dataset.json
  done
  cp data/eval2000.json data/backupeval2000.json
  cp data/train.json data/backuptrain.json
  python3 ../core/json_builder.py -i data/backupeval2000.json -o data/eval2000.json -b 50 -s 800
  python3 ../core/json_builder.py -i data/backuptrain.json -o data/train.json -b 50 -s 800
fi

#python3 test.py
#exit 0
if [ $stage -le 4 ]; then
  echo "Stage 4: Extraction of activation vectors per layer per epoch"
  [ ! -d $KALDI_ROOT ] && echo "Expected $KALDI_ROOT to exist" && exit 1;
  for check in $experiment; do
    if [[ $check == *"checkpoint"* ]]; then
      echo $check
      ln -sf $check /share/mini1/sw/spl/espresso/new_svcca/svcca/exp/checkpoints/
    fi
  done
  
  #rm /share/mini1/sw/spl/espresso/new_svcca/svcca/exp/checkpoints/checkpoint_*
  #printenv | less
  files="/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/checkpoints/*"
  embeddings="/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/tmp_embeddings"
  mkdir -p /share/mini1/sw/spl/espresso/new_svcca/svcca/exp/all_epoch_embeddings/$name
  fin="/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/all_epoch_embeddings/$name"
  for f in $files; do
    echo "Processing $f file..."
    model_file=$(basename $f)
    echo $model_file
    ##
    #exp/svcca/model_parse.py --model-files $model_file
    ##
    embd=$embeddings/saved
    #touch $embd.lstm.embd.pkl
    #touch $embd.cnn.embd.pkl
    echo "Model file: $model_file"
    [ -f local/wer_output_filter ] && opts="$opts --wer-output-filter local/wer_output_filter"
    for dataset in $test_set; do
      decode_dir="/share/mini1/sw/spl/espresso/git20201130/examples/asr_swbd/exp/transformer_2021/decode"
      python3 ../espresso/speech_recognize.py data \
        --task speech_recognition_espresso --max-tokens 90000 --max-sentences 101 \
        --num-shards 2 --shard-id 0 --dict $dict --bpe sentencepiece --sentencepiece-model ${sentencepiece_model}.model \
        --non-lang-syms $nlsyms --gen-subset $dataset --max-source-positions 9999 --max-target-positions 999 \
        --path $model_file --beam 15 --max-len-a 0.1 --max-len-b 0 --lenpen 1.0 \
        --results-path $decode_dir $opts

      echo "log saved in ${decode_dir}/decode.log"
     
    done
    
    mkdir $fin/$model_file.dir
    mv $embeddings/* $fin/$model_file.dir
  done
fi

exit 0
svcca_dir='/share/mini1/sw/spl/espresso/new_svcca/svcca/exp/svcca/'
#python3 test.py
if [ $stage -le 5 ]; then
  echo "SVCCA Test"
  CUDA_VISIBLE_DEVICES=$free_gpu $svcca_dir/calculate.py
fi
exit 0

if [ $stage -le 6 ]; then
  echo "Generating PWCCA"
  CUDA_VISIBLE_DEVICES=$free_gpu $svcca_dir/pwcca_calculate.py
fi

