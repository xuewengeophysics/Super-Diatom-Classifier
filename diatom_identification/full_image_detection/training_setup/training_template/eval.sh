PIPELINE_CONFIG_PATH=$PWD"/$1pipeline$3.config"
CHECKPOINT_DIR=$PWD"/$1training/"
MODEL_DIR=$PWD"/$1eval/"
SAMPLE_1_OF_N_EVAL_EXAMPLES=$2
python ~/nvme/libs/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --run_once="True" \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
