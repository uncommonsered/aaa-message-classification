# Классификация сообщений в мессенджере на оскорбление/харасмент/угрозу или другой тип сообщений

# О проекте :

**Задача:** По тексту сообщения пользователя нужно определить, является ли сообщения оскорблением/харасментом/угрозой или нет. Это нужно для того, чтобы ограничивать отправку таких сообщений пользователям, тем самым улучшая общение внутри Авито.

## Структура репозитория 

```text
.
├── data/
│   ├── custom_augmentations.py
│   ├── data_labeling.ipynb
│   ├── llm_tools.py
│   └── regex_for_labeling.py
├── model/
│   ├── pretrained_models/
│   │   ├── rubert_harassment_model/
│   │   └── model.joblib
│   └── train_model.ipynb
├── server/
│   ├── main.py
│   └── requirements.txt
├── .gitignore
├── Dockerfile
├── README.md
├── docker-compose.yml
├── metrics_and_expected_solution
├── metrics_test.ipynb
├── models_and_evaluation.md
└── project_overview.md
```

Папка `data` содержит ноутбук с предразметкой сообщений и аугментацию. В файлы `.ру` вынесена часть кода с функциями для разметки\аугментации\фильтрации \
Папка `model` содержит папку с обученными моделями `pretrained_models` и ноутбук с первым обучением моделей \
Папка `server` содержит необходимый код для запуска сервиса \
Ноутбук `metrics_test.ipynb` содержит тесты для проверки latency при запущенном локально сервере \
 

## Запуск и использование

```bash
docker compose up -d
```

### Проверка сервиса

```bash
curl http://localhost:8000/health
```
Должен вернуться ответ:
```bash
{
        "status": "ok",
        "bert_loaded": True,
        "logreg_loaded": True
}
```

### Примеры использования 
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Привет!"}'
```

Возвращает json вида ```{'label' : ...}``` c одним из классов.

## Метрики

Используются:

- Latency (P50/P90/P98) 
- Throughput

Ноутбук с замером метрик находится в этом же репозитории (metrics_test.ipynb)
После запуска в docker контейнере, ноутбук может запускаться локально для расчета метрик.

# Участники :

**Название команды :** Градиентный спуск в депрессию  

**Состав команды :** Савицкий Владислав, Ангелина Букина
 
