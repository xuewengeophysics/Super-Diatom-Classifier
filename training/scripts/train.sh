# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/home/GTL/pfauregi/Training/setup01/models/model/faster_rcnn_resnet101.config"
MODEL_DIR="/home/GTL/pfauregi/Training/setup01/models/model"
NUM_TRAIN_STEPS=700
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python ~/venv/obj_dect/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
