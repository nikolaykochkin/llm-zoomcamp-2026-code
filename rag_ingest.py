from openai import OpenAI
from rag_helper import RAGBase
from ingest import load_faq_data, build_index
from dotenv import load_dotenv
import os
load_dotenv()


documents = load_faq_data()
index = build_index(documents)

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
)

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
)

answer = assistant.rag("I just discovered the course. Can I join now?")
print(answer)

print(assistant.rag("How do I get a certificate?"))
print(assistant.rag("Can I still join the course after it started?"))
