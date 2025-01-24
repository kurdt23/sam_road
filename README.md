## Инструкция для запуска на кластере
### Предподготовка
Перейти в папку ```_scratch2``` и склонировать репозиторий
```
cd _scratch2
git clone https://github.com/kurdt23/sam_road.git
```

Создать виртуальное окружение
```
python3.9 -m venv segm_models
source segm_models/bin/activate
```

Перейти в папку проекта, где ```$user``` имя вашего пользователя на кластере
```
cd /home/$user/_scratch2/sam_road/
```
> p.s. узнать свое имя на кластере
> ```whoami```

Выполнить запуск скрипта для установки модели, датасета и библиотек
```
bash install.sh
```

После завершения установочного скрипта, перейдите по пути и всё закомментируйте в ```__init__.py```
```
cd /misc/home6/$user/sam_road/sam/segment_anything/modeling/__init__.py
```

Перейдите по пути и закомментируйте в ```conftest.py```, раздел связанный с ```nx-loopback```
```
cd /misc/home6/$user/sam_road/segm_models/lib64/python3.9/site-packages/networkx/conftest.py
```
![image](https://github.com/kurdt23/sam_road/assets/148371058/6f321023-c285-47f9-bb96-52aede68fc6c)

### Регистрация на wandb
1. Перейдите на сайт https://wandb.ai/
2. Войдите в свою учетную запись или создайте новую, если у вас ее еще нет (можно через гитхаб войти и указать цель создания аккаунта для учебы)
3. После входа в систему, наведите курсор мыши на свой профиль в правом верхнем углу страницы и выберите "Настройки" или "Settings".
4. В разделе "Profile" или "Профиль" вы найдете ваш API ключ.
5. В консоле введите ```wandb login``` и далее свой API ключ.
```
wandb login
```

> В дальнейшем на этом сайте, можно отслеживать результаты обучения по графикам, перейдя по ссылке из файла с логами, пример:
>
>  ![image](https://github.com/kurdt23/sam_road/assets/148371058/deb794b2-1efd-49d5-b03e-dacbd8c00fb4)



### Запуск обучения
Перейти в папку проекта, включить виртуальное окружение и запустить тренировочный скрипт
```
cd /home/$user/_scratch2/sam_road/
python3.9 -m venv segm_models
source segm_models/bin/activate
bash train_sbatch.sh
```
[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&duration=3000&pause=2000&color=F711B4&random=false&width=1200&lines=%D0%95%D1%81%D0%BB%D0%B8%2C+%D0%B2%D1%81%D0%B5+%D1%81%D0%B4%D0%B5%D0%BB%D0%B0%D0%BB%D0%B8+%D0%B2%D0%B5%D1%80%D0%BD%D0%BE+%D0%B8+%D0%B2+%D0%BB%D0%BE%D0%B3%D0%B0%D1%85+%D0%BF%D0%BE%D1%8F%D0%B2%D0%B8%D0%BB%D0%B8%D1%81%D1%8C+%D1%81%D1%82%D1%80%D0%BE%D0%BA%D0%B8+%D1%81+%D1%8D%D0%BF%D0%BE%D1%85%D0%B0%D0%BC%D0%B8%2C+%D0%BF%D0%BE%D0%B7%D0%B4%D1%80%D0%B0%D0%B2%D0%BB%D1%8F%D1%8E!+You're+breathtaking!+%E2%9C%A8)](https://git.io/typing-svg)


----------------------------------------------------------------------

### Для пользовательского датасета

Файлы с подписью ```ekb```. Причем с подписью ```ekb2``` это RG1, а просто ```ekb``` это RG2.

Для RG2:
 ```
ekb/2048
ekb/400
 ```
Сделать ссылки на картинки:
![image](https://github.com/user-attachments/assets/c92dca3f-8a32-43d2-8054-2f03301f0dd5)


Для RG1:

 ```
ekb/ekb2
 ```
Сделать ссылки на картинки:
![image](https://github.com/user-attachments/assets/45e05af9-fc04-4dcc-b0e0-61e3502b17a3)

Конфиги смотреть в папке ```config``` с соответсвующими подписями для запуска нужного города и разрешения.

### Semi-supervised learning

Файлы с подписью ```semi``` для запуска дообучения с частичным привлечением учителя.

----------------------------------------------------------------------

# Official codebase for "Segment Anything Model for Road Network Graph Extraction", CVPRW 2024
https://arxiv.org/pdf/2403.16051.pdf

The paper has been accepted by IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR) 2024, 2nd Workshop on Scene Graphs and Graph Representation Learning.

## Demos
Predicted road network graph in a large region (2km x 2km).
![sam_road_cover](imgs/sam_road_cover.png)

Predicted road network graphs and corresponding masks in dense urban with complex and irregular structures.
![sam_road_mask_and_graph](imgs/sam_road_mask_and_graph.png)

## Installation
You need the following:
- an Nvidia GPU with latest CUDA and driver.
- the latest pytorch.
- pytorch lightning.
- wandb.
- Go, just for the APLS metric (we should really re-write this with pure python when time allows).
- and pip install whatever is missing.


## Getting Started

### SAM Preparation
Download the ViT-B checkpoint from the official SAM directory. Put it under:  
-sam_road  
--sam_ckpts  
---sam_vit_b_01ec64.pth  

### Data Preparation
Refer to the instructions in the RNGDet++ repo to download City-scale and SpaceNet datasets.
Put them in the main directory, structure like:  
-sam_road  
--cityscale  
---20cities  
--spacenet  
---RGB_1.0_meter  

and run python generate_labes.py under both dirs.

### Training
City-scale dataset:  
python train.py --config=config/toponet_vitb_512_cityscale.yaml  

SpaceNet dataset:  
python train.py --config=config/toponet_vitb_256_spacenet.yaml  

You can find the checkpoints under lightning_logs dir.

### Inference
python inferencer.py --config=path_to_the_same_config_for_training --checkpoint=path_to_ckpt  
This saves the inference results and visualizations.

### Test
Go to cityscale_metrics or spacenet_metrics, and run  
bash eval_schedule.bash  

Check that script for details. It runs both APLS and TOPO and stores scores to your output dir.

## Citation
```
@article{hetang2024segment,
  title={Segment Anything Model for Road Network Graph Extraction},
  author={Hetang, Congrui and Xue, Haoru and Le, Cindy and Yue, Tianwei and Wang, Wenping and He, Yihui},
  journal={arXiv preprint arXiv:2403.16051},
  year={2024}
}
```

## Acknowledgement
We sincerely appreciate the authors of the following codebases which made this project possible:
- Segment Anything Model  
- RNGDet++  
- SAMed  
- Detectron2  

## TODO List
- [x] Basic instructions
- [x] Organize configs
- [ ] Add dependency list
- [x] Add demos
- [ ] Add trained checkpoints



