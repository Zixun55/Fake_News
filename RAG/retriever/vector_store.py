#%%
# csv loader  
from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader('./knowledge_base/clean_data/politifact_all.csv')

politifact_data = loader.load()

#%%
# text splitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=5)

# all_splits = text_splitter.split_documents(politifact_data)

# %%
# embedding model
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# %%
# vector database
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

vector_store = FAISS.from_documents(
    documents=politifact_data,
    embedding=embedding,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

vector_store.save_local('./politifact_all')