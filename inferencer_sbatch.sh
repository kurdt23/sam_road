#!/bin/sh
export CUDA_HOME='/opt/cuda-12.4'
module load gpu/cuda-12.4

source segm_models/bin/activate

# ����� ������������ batch_size �� �������� � ��������
#for (( i=50; i > 40; i=i-1 ))
#do

CPU_TRAIN=true
#BATCH=$i
BATCH=16
EPOH=10
IMAGE_SIZE="512"

# �� �������, ���������� ��� ������������ ����� cpu � gpu

SBATCH_CPU=""
PYTHON_CPU=""

if [ $CPU_TRAIN == false ]
then
  SBATCH_CPU="-p v100 --gres=gpu:v100:1 --nodelist=tesla-v100"
else
  PYTHON_CPU="--cpu"
fi

# ��� �������� ���������� � ���� ./logs/"_%j" ��� j - ����� ������ �� �������� 
#����������: ����� ����� wrap ��������� ���
# ������ � ��������� ��������

# -w "tesla-a101" \
# �gres=gpu:a100/v100:1

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
#���� ��� ���� ������ � ������� ��������� ������ �� ��� ���� �� ���������� ���� �� CPU, �������� ��� ���������
#  ������ ����� ����� sbatch � ������� � ������ ����� ������� ������ �� CPU
# -p debug \
#-t 00:30:00 --workers 1
# �������� ����� num_workers ���������� ��� � warnings � ����� ��������
