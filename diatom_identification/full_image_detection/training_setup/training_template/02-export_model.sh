# From tensorflow/models/research/
if [[ $2 -eq 0 ]] ; then
    echo 'EXITING. Please provide some checkpoint : eg. ./export_model.sh folder/ 523'
    exit 0
fi
if [ -z "$(ls -A ./$1export)" ]; then
   echo "Exporting model from checkpoint $2"
else
   echo "EXITING. Export directory not empty."
   exit 0
fi
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH=$PWD"/$1pipeline$3.config"
TRAINED_CKPT_PREFIX=$PWD"/$1training/model.ckpt-$2"
EXPORT_DIR=$PWD"/$1export/"
python ~/nvme/libs/models/research/object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
