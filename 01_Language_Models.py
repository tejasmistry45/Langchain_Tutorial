import getpass
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API Key for GROQ: ")


model = ChatGroq(model="openai/gpt-oss-120b")

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"language": "French", "text": "Hello"})

response = model.invoke(prompt)
print(response.content)