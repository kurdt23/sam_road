#!/bin/bash

#подключение к виртуальному пространству
python3.9 -m venv segm_models
source segm_models/bin/activate

# установка библиотек
python -m pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html

# скачивание модели
mkdir sam_ckpts
cd sam_ckpts
wget -c https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd

# скачивание датасета
cp -r /home/$user/_scratch2/sam_road/cityscale/process_data.sh /misc/home6/m_imm_freedata/Segmentation/Projects
cd /misc/home6/m_imm_freedata/Segmentation/Projects
bash process_data.sh
ln -s /misc/home6/m_imm_freedata/Segmentation/Projects/20cities /home/$user/_scratch2/sam_road/cityscale/

# клонирование дополнительных репозиториев для запуска обучения
cd /home/$user/_scratch2/sam_road/
rm -r sam
git clone https://github.com/htcr/segment-anything-road.git
mv /home/$user/_scratch2/sam_road/segment-anything-road /home/$user/_scratch2/sam_road/sam
cd /home/$user/_scratch2/sam_road/
git clone https://github.com/htcr/segment-anything-road.git

# Подготовка данных
cd /home/$user/_scratch2/sam_road/cityscale
sbatch --mem=32000 -t 10:00:00 --output='%j' --wrap="python3 generate_labels.py"

# создание папки для логов
cd /home/$user/_scratch2/sam_road/
mkdir logs

