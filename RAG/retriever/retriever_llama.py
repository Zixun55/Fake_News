from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector = FAISS.load_local(
    folder_path='./politifact_all',
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

query = "What is Kelly Ayotte's stance on abortion and reproductive rights?"

query_vector = embedding.embed_query(query)

retrieved_docs = vector.similarity_search_with_score_by_vector(query_vector, k=3)

for i, (doc, score) in enumerate(retrieved_docs):
    print(f'Top {i+1} Result (Score: {score}):\n{doc.page_content}\n')