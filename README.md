Материалы для трёх статей по оптимизаторам.

Необходимо предварительно установить солверы, как описано ниже.
Наиболее простой способ установки необходимых пакетов для оптимизаторов осуществляется
через пакетный менеджер conda(идет в составе [Anaconda](https://www.anaconda.com/) 
или [Miniconda](https://docs.conda.io/en/latest/miniconda.html))


### Установка через conda
Пример сборки:

1. Создать окружение 
```
conda create --name opt_conda_env python=3.8
```
2. Запуск окружения
```
conda activate opt_conda_env
```
3. Установка необходимых пакетов
```
conda install -c conda-forge conda_requirements.txt
```
4. Добавление окружения в jupyter
```
python -m ipykernel install --user --name=opt_conda_env
```


### Запуск в докере

1. Собрать контейнер из Dockerfile с тегом opt из текущей директории:
```
docker build -t opt .
```

2. Запустить контейнер с mount текущей директории <-> контейнер:
```
docker run -dp 3000:3000 -w /app -v "$(pwd):/app" -i -t opt
```

Запуск контейнера с возможностью запустить jupyter lab
```
docker run -dp 8888:8888 -w /app -v (pwd):/app -i -t opt 
```

3. Подставить CONTAINER ID (из команды docker ps) в команду:
```
docker attach <CONTAINER ID>
```

4. Далее, в контейнере запустить расчёты:
```
python runner.py
```

5. Запуск jupyter lab в контейнере
```
jupyter-lab --ip=0.0.0.0 --no-browser --allow-root
```

6. Выйти из контейнера: 
```
exit
```

