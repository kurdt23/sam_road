#!/bin/sh
export CUDA_HOME='/opt/cuda-12.4'
module load gpu/cuda-12.4

source segm_models/bin/activate

# можно поперебирать batch_size от большего к меньшему
#for (( i=50; i > 40; i=i-1 ))
#do

CPU_TRAIN=false
#BATCH=$i
#BATCH=64
#EPOH=30
#IMAGE_SIZE="256"

# Не трогать, переменные для переключения между cpu и gpu

SBATCH_CPU=""
PYTHON_CPU=""

if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p v100 --gres=gpu:v100:1 --nodelist=tesla-v100"
else
  PYTHON_CPU="--cpu"
fi

# лог обучения сохранится в файл ./logs/"_%j" где j - номер задачи на кластере 
#ПРИМЕЧАНИЕ: после слова wrap вставлять все
# строки в одинарных кавычках

# -w "tesla-a101" \
# —gres=gpu:a100/v100:1

sbatch \
-w "tesla-a101" \
--cpus-per-task=8 \
--mem=45000 \
$SBATCH_CPU \
-t 20:00:00 \
--job-name=segm \
--output=./logs/train"_%j" \
\
--wrap="python inferencer.py --config=config/toponet_vitb_256_spacenet.yaml --checkpoint=lightning_logs/mac62ayw/checkpoints/epoch=29-step=39690.ckpt"