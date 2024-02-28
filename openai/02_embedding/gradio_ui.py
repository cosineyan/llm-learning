import gradio as gr

from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI

import os
from dotenv import load_dotenv


def initialize_data(vector_store_dir: str="data/amazon-food-reviews-faiss"):
    db = FAISS.load_local(vector_store_dir, AzureOpenAIEmbeddings())
    llm = AzureChatOpenAI(model_name="gpt-35-turbo", temperature=0.5)
    
    global AMAZON_REVIEW_BOT    
    AMAZON_REVIEW_BOT = RetrievalQA.from_chain_type(llm,
                  retriever=db.as_retriever(search_type="similarity_score_threshold",
                    search_kwargs={"score_threshold": 0.7}))
    AMAZON_REVIEW_BOT.return_source_documents = True

    return AMAZON_REVIEW_BOT

def chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    enable_chat = True

    ans = AMAZON_REVIEW_BOT({"query": message})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    else:
        return "I don't know."
    

def launch_ui():
    demo = gr.ChatInterface(
        fn=chat,
        title="Amazon Food Review",
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://pvg-azure-openai-uk-south.openai.azure.com/openai"
    env_path = os.getenv("HOME") + "/Documents/src/openai/.env"
    load_dotenv(dotenv_path=env_path, verbose=True)
    
    initialize_data()
    launch_ui()
