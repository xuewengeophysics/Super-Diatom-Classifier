PIPELINE_CONFIG_PATH=$PWD"/$1pipeline$3.config"
MODEL_DIR=$PWD"/$1training/"
NUM_TRAIN_STEPS=$2
SAMPLE_1_OF_N_EVAL_EXAMPLES=$4
python -W ignore ~/nvme/libs/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
