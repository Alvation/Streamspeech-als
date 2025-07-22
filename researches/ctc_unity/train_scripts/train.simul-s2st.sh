export CUDA_VISIBLE_DEVICES=0,1,2,3

LANG=de
DATA_ROOT=/workspace/data/cvss/cvss-c
DATA=$DATA_ROOT/${LANG}-en/fbank2unit
model=streamspeech.simul-s2st.${LANG}-en

fairseq-train $DATA \
  --user-dir researches/ctc_unity \
  --config-yaml config_gcmvn.yaml --multitask-config-yaml config_mtl_asr_st_ctcst.yaml \
  --task MMS_LLaMA_training --target-is-code --target-code-size 1000 --vocoder code_hifigan  \
  --criterion speech_to_unit_2pass_ctc_asr_st --label-smoothing 0.1 --rdrop-alpha 0.0 \
  --arch streamspeech --share-decoder-input-output-embed \
  --k1 0 --k2 0 --n1 1 --n2 -1 \
  --multichunk \
  --dropout 0.1 --attention-dropout 0.1\
  --train-subset train --valid-subset dev \
  --save-dir researches/ctc_unity/checkpoints/$model \
  --validate-interval 1000 --validate-interval-updates 1000 \
  --save-interval 1 --save-interval-updates 1000 \
  --keep-last-epochs 15 \
  --no-progress-bar --log-format json --log-interval 100 \
  --lr 0.001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-7 --warmup-updates 10000 \
  --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 1.0 \
  --max-tokens 22000 --max-target-positions 1200 --update-freq 2 \
  --keep-interval-updates 40 \
  --keep-best-checkpoints 20 \
  --seed 1 --fp16 --num-workers 8 