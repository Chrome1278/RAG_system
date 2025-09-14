# RAG_system


# Структура проекта

- _data/_ - исходные данные и база индексов.
- _notebooks/preprocess_data.ipynb_ - анализ и предобработка исходных данных.
- _notebooks/data_chunking.ipynb_ - формирование базы индексов для векторного поиска.
- _src/search_engine.py_ - модуль для поиска ближайших батчей по заданному запросу.
- _src/answer_generator.py_ - модуль для получения ответа от LLM по заданному вопросу.
- _web_app.py_ - приложение для запуска RAG системы.


# Запуск веб-интерфейса с RAG системой

1. Загрузка зависимостей проекта:
```
pip install -r ./requirements.txt 
```
2. Формирование базы индексов movies_info.index (если её нет) через _notebooks/data_chunking.ipynb_


3. Запуск веб-интерфейса:
```
streamlit run web_app.py
```