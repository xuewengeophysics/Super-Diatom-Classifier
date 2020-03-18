# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH="/home/GTL/pfauregi/Training/setup02/models/model/config/mask_rcnn_inception_resnet_v2_atrous_coco.config"
MODEL_DIR="/home/GTL/pfauregi/Training/setup02/models/model"
NUM_TRAIN_STEPS=700
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python -W ignore ~/venv/obj_dect/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
