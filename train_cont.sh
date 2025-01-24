#!/bin/sh
export CUDA_HOME='/opt/cuda-12.4'
module load gpu/cuda-12.4

source segm_models/bin/activate

# можно поперебирать batch_size от большего к меньшему
#for (( i=50; i > 40; i=i-1 ))
#do

CPU_TRAIN=false
#BATCH=$i
BATCH=64
EPOH=50
IMAGE_SIZE="512"

# Не трогать, переменные для переключения между cpu и gpu

SBATCH_CPU=""
PYTHON_CPU=""

if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p hiperf --gres=gpu:1 --nodelist=tesla-v100"
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
--cpus-per-task=12 \
--mem=40000 \
$SBATCH_CPU \
-t 20:00:00 \
--job-name=segm \
--output=./logs/train_continue"_%j" \
\
--wrap="echo 'python train.py --config=config/config/toponet_vitb_512_cityscale.yaml --resume=lightning_logs/gvdsoqxd/checkpoints/epoch=9-step=25000.ckpt' && python train.py --config=config/toponet_vitb_512_cityscale.yaml --resume=lightning_logs/gvdsoqxd/checkpoints/epoch=9-step=25000.ckpt"

#done
#Если все узлы заняты и хочется запустить задачу на пол часа на отладочном узле на CPU, вставить две следующие
#  строки после слова sbatch и указать в начале этого скрипта работу на CPU
# -p debug \
#-t 00:30:00 --workers 1
# смотрите какое num_workers рекоменуют вам в warnings в логах обучения
