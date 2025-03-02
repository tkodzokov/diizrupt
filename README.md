# 🤖 diizrupt AI помощник риск-менеджера

### В основу решения легла архитектура Multi-agent supervisor https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/ 
![image](https://github.com/user-attachments/assets/300b38a4-f7f0-4c52-a0e3-bb7e191a8644)

### Основной инструмент для решения задачи - библиотека langgraph-supervisor-py

langgraph-supervisor-py - оболочка для LangGraph, которая кардинально упрощает создание систем типа Multi-agent supervisor.
Появилась через день после анонса хакатона, первая версия без критичных багов - за 5 дней до дедлайна

Реализована следующая схема взаимодействия:
[schema.png
](https://github.com/tkodzokov/diizrupt/blob/main/schema.png)

Воспользоваться помощником можно с помощью ноутбука https://github.com/tkodzokov/diizrupt/blob/main/___AI%20%D0%BF%D0%BE%D0%BC%D0%BE%D1%89%D0%BD%D0%B8%D0%BA.ipynb

Также взаимодействовать с AI помощником можно с помощью telegram-бота @...

Само решение в https://github.com/tkodzokov/diizrupt/blob/main/agents.py

Примеры успешного взаимодейтвия с помощником - https://github.com/tkodzokov/diizrupt/blob/main/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D1%8B%20%D1%83%D1%81%D0%BF%D0%B5%D1%85%D0%B0.ipynb 
