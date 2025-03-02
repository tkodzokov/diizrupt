auth_project='=='
auth_personal='=='


from langchain_gigachat import GigaChat
from langchain_core.messages import HumanMessage, SystemMessage

model_lite = GigaChat(credentials=auth_personal, verify_ssl_certs=False,
               profanity_check=False, model='GigaChat',
               temperature=0.00)
model_pro = GigaChat(credentials=auth_project, verify_ssl_certs=False,
               profanity_check=False, model='GigaChat-Pro',
               temperature=0.00)
model_max = GigaChat(credentials=auth_project, verify_ssl_certs=False,
               profanity_check=False, model='GigaChat-Max',
               temperature=0.00)
llm_db = GigaChat(credentials=auth_project, verify_ssl_certs=False,
               profanity_check=False, model='GigaChat-Max',
               temperature=0.00)

from langchain_core.prompts import PromptTemplate

template = '''You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run,
then look at the results of the query and return the answer.

Выводи в ответе ТОЛЬКО SQL запрос.

Если не уточняется, выведи первые {top_k} строк.

Only use the following tables:
{table_info}.

Используй следующую информацию о таблицах:
application: информация о заявках на ипотеку
client: информация о клиентах, подававших заявки на ипотеку. Клиенты с client_num=0 - основные заемщики. 1, 2 и т.д. - созаемщики
decision: информация о принятом решении по кредиту. 1 - ипотека одобрена, 0 - отклонена
product: справочник категорий ипотечных продуктов
subproduct: справочник подкатегорий ипотечных продуктов
application_category: к какой категории относятся заемщики
channel: справочник каналов, по которым пришел заемщик
reject_rule: справочник причин отказов
territory_bank: справочник территориальных банков

Учитывай, что на вопросы, связанные с одобрением заявок необходимо делать join на таблицу decision.
Если необходимо, можешь использовать следующие формулы для расчета:
PTI = (liability + payment) / income
LTV = 1 - initial_payment_rate

AR это approve rate, кол-во одобренных / кол-во всех заявок

Question: {input}'''

tables = [
    "application",
    "client",
    "decision",
    "product",
    "subproduct",
    "application_category",
    "channel",
    "reject_rule",
    "territory_bank"
]

prompt_db = PromptTemplate.from_template(template, table_info=tables)

db_path = 'db_mortgage.db'
import sqlite3

conn = sqlite3.connect(db_path)
cur = conn.cursor()

from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///" + db_path)
from langchain.chat_models.gigachat import GigaChat


from langchain.chains import create_sql_query_chain
chain_db = create_sql_query_chain(llm_db, db, prompt=prompt_db)

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

def database_func(user_question: str) -> str:
    """На основе заданного вопроса сформируй SQL запрос, выполни и ответь на поставленный вопрос"""
    print('+'*100)
    print(user_question)
    response = chain_db.invoke({"question": user_question})
    print(response)

    response_table = db.run(response[7:-5], include_columns=True)
    
    print(response_table)
    
    aaa = model_pro.invoke(
        "Я приведу тебе: " + 
        "1. Вопрос пользователя " + 
        "2. SQL запрос, который был сформирован чтобы ответить на вопрос пользователя. " + 
        "3. Ответ базы данных на этот SQL запрос. " + 
        "Сопоставь вопрос пользователя с ответом ответ БД и напиши текстом, как бы человек ответил на изначально поставленный запрос пользователя. " + 
        "Итак: " + 
        "1. Вопрос пользователя: " + user_question + 
        "2. SQL запрос: " + response[7:-5] + 
        "3. Ответ базы данных: " + db.run(response[7:-5]) + 
        " . Приведи только ответ пользователя от его лица"
        )
    
    print(aaa.content)
    # df = pd.read_sql_query(response[7:-5], conn)
    # print(df)
    print('+'*100)
    answer = 'Готово: ' + aaa.content + '. Данные на которых получен ответ: ' + response_table
    print(answer)
    return answer

database_agent = create_react_agent(
    model=model_pro,
    tools=[database_func],
    name="database_expert",
    # prompt="You are agent who interact with Database to answer questions. ALWAYS use tool for tasks"
    prompt="""Ты агент, который взаимодействует с Базой данных по ипотечным заявкам для ответа на пользовательские вопросы. 
    ВСЕГДА используй для работы tools. Результат работы tools возвращай агенту-руководителю ответ целиком как есть, НЕ СОКРАЩАЙ ОТВЕТ и не добавляй ничего"""

)

analyst_agent = create_react_agent(
    model=model_pro,
    tools=[],
    name="analyst_expert",
    # prompt="You are agent who interact with Database to answer questions. ALWAYS use tool for tasks"
    prompt="""Ты высококласный аналитик, который анализирует данные. Ты должен использовать ТОЛЬКО те данные, которые ранее были получены в диалоге."""
)

# Create supervisor workflow
workflow = create_supervisor(
    [database_agent, analyst_agent],
    model=model_max,
    prompt=(
        "Ты агент-руководитель, который управляет агентами database_expert и analyst_expert. "
        "НИ В КОЕМ СЛУЧАЕ не бращайся к database_agent или analyst_expert БОЛЕЕ ОДНОГО РАЗА"
        
        "Ключевые фразы для database_agent: 'сколько', 'какой процент', 'найди', 'выведи', ' ответь в разрезе' "
        "Ключевые фразы для analyst_agent: 'проанализируй', 'аномалии', 'сделай выводы' "
        
        "Для агента database_agent перенаправляй пользовательские вопросы целиком как есть. "
        "Результат работы database_agent СРАЗУ верни пользователю целиком как есть, сам ничего не выдумывай, не добавляй и не отрезай от ответа"
        "НИ В КОЕМ СЛУЧАЕ не обращайся к database_expert более одного раза"

        "Для агента analyst_agent перенаправляй пользовательские вопросы целиком как есть. "
        "Результат работы analyst_agent СРАЗУ верни пользователю целиком как есть, сам ничего не выдумывай, не добавляй и не отрезай от ответа"

        "Заверши workflow, после того как какой-нибудь агент скажет 'Готово ...' "
        
    )
)

app = workflow.compile()

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

checkpointer = InMemorySaver()
# store = InMemoryStore()




def call_bot(string, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    app = workflow.compile(
        checkpointer=checkpointer
    )
    result = app.invoke({
        "messages": [
            {
                "role": "user",
                "content": string
            }
        ]
    }, config=config)
    return result
