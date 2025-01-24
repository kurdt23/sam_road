#!/bin/sh
export CUDA_HOME='/opt/cuda-12.4'
module load gpu/cuda-12.4

source segm_models/bin/activate

# можно поперебирать batch_size от большего к меньшему
#for (( i=50; i > 40; i=i-1 ))
#do

CPU_TRAIN=true
#BATCH=$i
BATCH=16
EPOH=10
IMAGE_SIZE="512"

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
-p apollo \
--mem=45000 \
$SBATCH_CPU \
-t 10:00:00 \
--job-name=segm \
--output=./logs/inferencer"_%j" \
\
--wrap="python inferencer.py --config=config/toponet_vitb_512_ekb.yaml --checkpoint=lightning_logs/vhfsw197/checkpoints/epoch=9-step=25000.ckpt"

#done
#Если все узлы заняты и хочется запустить задачу на пол часа на отладочном узле на CPU, вставить две следующие
#  строки после слова sbatch и указать в начале этого скрипта работу на CPU
# -p debug \
#-t 00:30:00 --workers 1
# смотрите какое num_workers рекоменуют вам в warnings в логах обучения
