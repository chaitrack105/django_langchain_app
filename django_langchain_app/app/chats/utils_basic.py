from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
YOUR_GOOGLE_API_KEY = os.getenv('YOUR_GOOGLE_API_KEY')

#llm= ChatOpenAI(model='gpt-4o-mini', openai_api_key=OPENAI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=YOUR_GOOGLE_API_KEY)

system_prompt = SystemMessage(content="You are an helpful assistant")


def generate_response(user_input, history):
    history_msgs = []
    for msg in history:
        if msg.sender == 'human':
            history_msgs.append(HumanMessage(content=msg.text))
        elif msg.sender == 'ai':
            history_msgs.append(AIMessage(content=msg.text))

    history_msgs.append(AIMessage(content=user_input))

    prompt = ChatPromptTemplate.from_messages([system_prompt] + history_msgs)
    chain = prompt | llm | StrOutputParser()  

    return chain.invoke({})