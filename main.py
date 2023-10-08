from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == "__main__":
    pdf_path = "/Users/sanjaysingh/langchain/vecto-db-intro/ResearchPaper.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=document)

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents=document, embedding=embeddings)
    vector_store.save_local("faiss_index_react")

    new_vector_store = vector_store.load_local("faiss_index_react", embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vector_store.as_retriever())

    response = qa.run("Exaplain ReAct idea inception?")
    print(f"Response: {response}")




# import os
#
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain import VectorDBQA, OpenAI
#
# import pinecone
#
# pinecone.init(api_key="b6bfb032-c5a4-4724-a6cd-31efebd3fb25", environment="us-west1-gcp-free")
#
# if __name__ == "__main__":
#     print("Hello Vectordatabase !!!")
#
#     loader = TextLoader("/Users/sanjaysingh/langchain/vecto-db-intro/mediumblogs/medium.txt")
#     document = loader.load()
#
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#     texts = text_splitter.split_documents(document)
#     print(len(texts))
#
#     embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
#     docsearch = Pinecone.from_documents(
#         texts, embeddings, index_name="medium-blogs-embeddings-index"
#     )
#
#     qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
#     query = "Is there a docker version of vector database?"
#
#     result = qa({"query": query})
#
#     print(result)
