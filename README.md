# captcha
 Сaptcha solution
 
# Тренировка (лучшая конфигурация)
  - В качестве модели был выбран resnet34 с заморозкой 50 слоев
  - optimizer -> Adam(lr = 1e-3)
  - loss -> MultiLabelSoftMarginLoss()
  - scheduler -> StepLR(step_size=25, gamma=0.1)
  
 
# Описание файлов
  * train_model.ipynb - тренировка на google colab (gpu)
  * main.py - файл для получения результата на тестируемом датасете
  * dataload.py - обработка данных

# main.py
Файл main.py может запускаться с параметрами:
  * --path - Обязательный параметр. Указывается имя папки, в которой лежат тестируемые данные. Папка должна лежать в корне проекта.
  * --weight_name - Не обязательный параметр, по умолчанию weight.pt. Имя файла весов. Этот файл должен лежать в корне проекта
  * --delete (n/y) - Не обязательный параметр, по умолчанию n. Удаляет все посторонние файлы, которые не имеют расширения png.

# Dockerfile
  ```
  docker build -t mati-test .
  docker run -t mati-test --path PATH_TEST_FOLDER --weight_name WEIGHT_NAME --delete y
  ```
