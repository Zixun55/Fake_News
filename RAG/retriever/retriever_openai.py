import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_openai import ChatOpenAI

#### 1. 讀取知識庫 ####
csv_file = "./knowledge_base/clean_data/politifact_clean.csv"
df = pd.read_csv(csv_file)

docs = df["Explanation"].astype(str).fillna("").tolist()

#### 2. 文本分割 ####
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.create_documents(docs)

#### 3. 建立向量資料庫 ####
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

#### 4. 檢索與生成回答 ####
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 測試查詢
question = "Mark Cuban moved company from Texas to California"
response = rag_chain.invoke(question)
print(response)
