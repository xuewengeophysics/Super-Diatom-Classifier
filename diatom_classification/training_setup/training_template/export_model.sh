# From tensorflow/models/research/
if [[ $1 -eq 0 ]] ; then
    echo 'EXITING. Please provide some checkpoint : eg. ./export_model.sh 523'
    exit 0
fi
if [ -z "$(ls -A ./models/model/export)" ]; then
   echo "Exporting model from checkpoint $1"
else
   echo "EXITING. Export directory not empty."
   exit 0
fi
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH="/home/GTL/pfauregi/Training/setup02/models/model/config/mask_rcnn_inception_resnet_v2_atrous_coco.config"
TRAINED_CKPT_PREFIX="//home/GTL/pfauregi/Training/setup02/models/model/model.ckpt-$1"
EXPORT_DIR="/home/GTL/pfauregi/Training/setup02/models/model/export"
python ~/venv/obj_dect/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
