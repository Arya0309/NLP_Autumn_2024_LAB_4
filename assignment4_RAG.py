#!/usr/bin/env python
# coding: utf-8

# # RAG using Langchain

# ## Packages loading & import

# In[1]:


# !pip install langchain
# !pip install langchain_community
# !pip install langchain_huggingface
# !pip install langchain_text_splitters
# !pip install langchain_chroma
# !pip install rank-bm25
# !pip install huggingface_hub


# In[2]:


import os
import json
import bs4
import nltk
import torch
import pickle
import numpy as np

# from pyserini.index import IndexWriter
# from pyserini.search import SimpleSearcher
from numpy.linalg import norm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import JinaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm


# In[3]:


nltk.download("punkt")
nltk.download("punkt_tab")


# ## Hugging face login
# - Please apply the model first: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# - If you haven't been granted access to this model, you can use other LLM model that doesn't have to apply.
# - You must save the hf token otherwise you need to regenrate the token everytime.
# - When using Ollama, no login is required to access and utilize the llama model.

# In[4]:


from huggingface_hub import login

hf_token = "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"
login(token=hf_token, add_to_git_credential=True)


# In[5]:


get_ipython().system('huggingface-cli whoami')


# ## TODO1: Set up the environment of Ollama

# ### Introduction to Ollama
# - Ollama is a platform designed for running and managing large language models (LLMs) directly **on local devices**, providing a balance between performance, privacy, and control.
# - There are also other tools support users to manage LLM on local devices and accelerate it like *vllm*, *Llamafile*, *GPT4ALL*...etc.

# ### Launch colabxterm

# In[6]:


# # TODO1-1: You should install colab-xterm and launch it.
# # Write your commands here.
# !pip install colab-xterm
# %load_ext colabxterm


# In[7]:


# # TODO1-2: You should install Ollama.
# # You may need root privileges if you use a local machine instead of Colab.
# !curl -fsSL https://ollama.com/install.sh | sh


# In[8]:


# %xterm


# In[9]:


# TODO1-3: Pull Llama3.2:1b via Ollama and start the Ollama service in the xterm
# Write your commands in the xterm


# In[10]:


# %xterm


# ## Ollama testing
# You can test your Ollama status with the following cells.

# In[11]:


# Setting up the model that this tutorial will use
MODEL = "llama3.2:1b"  # https://ollama.com/library/llama3.2:3b
EMBED_MODEL = "jinaai/jina-embeddings-v3"
"""
Model options:
jinaai/jina-embeddings-v2-base-en
jinaai/jina-embeddings-v3
pingkeest/learning2_model
all-MiniLM-L12-v2
bert-base-uncased
"""

VERSION = "BASIC"
"""
ADVANCED
BASIC
"""

DATA = "QA_10"  # "QA_100" or "QA_10"
"""
QA_100
QA_10
"""

STRICT = True

threshold = 0.6


# In[12]:


# Initialize an instance of the Ollama model
llm = Ollama(
    model=MODEL, temperature=0
)  # set temperature to 0 for deterministic results
# Invoke the model to generate responses
response = llm.invoke("What is the capital of Taiwan?")
print(response)


# ## Build a simple RAG system by using LangChain

# ### TODO2: Load the cat-facts dataset and prepare the retrieval database

# In[13]:


# !wget https://huggingface.co/ngxson/demo_simple_rag_py/resolve/main/cat-facts.txt


# In[14]:


# TODO2-1: Load the cat-facts dataset (as `refs`, which is a list of strings for all the cat facts)
# Write your code here

with open("cat-facts.txt", "r") as f:
    refs = f.readlines()


# In[15]:


from langchain_core.documents import Document

docs = [Document(page_content=doc, metadata={"id": i}) for i, doc in enumerate(refs)]


# In[16]:


# Create an embedding model
model_kwargs = {"trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": False}
embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


# In[17]:


# TODO2-2: Prepare the retrieval database
# You should create a Chroma vector store.
# search_type can be “similarity” (default), “mmr”, or “similarity_score_threshold”
vector_store = Chroma.from_documents(
    documents=docs,  # List of your document objects
    embedding=embeddings_model,  # Embedding model to encode the documents
)

retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
)


# ### Prompt setting

# In[18]:


# TODO3: Set up the `system_prompt` and configure the prompt.
if VERSION == "BASIC":
    system_prompt = """
    You are an AI assistant tasked with retrieving answers to user questions. Your answers must be derived strictly from the provided context and must match the wording in the context exactly.

    Context:
    {context}

    ### Guidelines:
    1. Use only the exact text from the provided context to answer the question. Do not infer, rephrase, or add information beyond what is explicitly stated.
    2. Answers must:
    - Match the exact wording and terminology from the context.
    - Include articles ("the," "a," etc.) and phrasing exactly as they appear in the context.
    3. For numerical answers or measurements, retain units and formatting exactly as shown in the context.
    4. If the context provides multiple valid answers, select only one of them.
    5. If the context contains no relevant information, respond with: "The answer is not available."
    6. Avoid elaboration, qualifiers, or dropping any part of the context text.

    Answer the following question accurately based on the context provided:
    """

if VERSION == "ADVANCED":
    if STRICT:
        system_prompt = """
        You are an AI assistant tasked with answering to user questions base on the retrieved context. Your answers must be derived strictly from the provided context.

        Context:
        {context}

        ### Guidelines:
        1. Provide only the core answer, without any additional text, explanation, or rephrasing.
        2. If the context contains no relevant information, respond with: "The answer is not available."

        Answer:
        """
    else:
        system_prompt = """
        You are an AI assistant tasked with answering to user questions base on the retrieved context.

        Context:
        {context}

        Answer:
        """


# Write your code here
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
print(prompt)


# - For the vectorspace, the common algorithm would be used like Faiss, Chroma...(https://python.langchain.com/docs/integrations/vectorstores/) to deal with the extreme huge database.

# In[19]:


# TODO4-1: Load the QA chain
# You should create a chain for passing a list of Documents to a model.
question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt,
)
# TODO4-2: Create the retrieval chain
# You should create retrieval chain that retrieves documents and then passes them on.
chain = create_retrieval_chain(
    retriever,
    question_answer_chain,
)


# In[20]:


# Question (queries) and answer pairs
# Please do not modify this cell.
queries = [
    "How much of a day do cats spend sleeping on average?",
    "What is the technical term for a cat's hairball?",
    "What do scientists believe caused cats to lose their sweet tooth?",
    "What is the top speed a cat can travel over short distances?",
    "What is the name of the organ in a cat's mouth that helps it smell?",
    "Which wildcat is considered the ancestor of all domestic cats?",
    "What is the group term for cats?",
    "How many different sounds can cats make?",
    "What is the name of the first cat in space?",
    "How many toes does a cat have on its back paws?",
]
answers = [
    "2/3",
    "Bezoar",
    "a mutation in a key taste receptor",
    ["31 mph", "49 km"],
    "Jacobson’s organ",
    "the African Wild Cat",
    "clowder",
    "100",
    ["Felicette", "Astrocat"],
    "four",
]


# In[21]:


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate(response, answer):

    embedding1 = model.encode(response)
    embedding2 = model.encode(answer)

    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])
    print(f"Semantic Similarity: {similarity[0][0]}")
    return similarity[0][0]


# In[24]:


import matplotlib.pyplot as plt
import json


counts = 0
similarity_scores = []
if VERSION == "ADVANCED":
    if DATA == "QA_100":

        with open("QA_100.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        for item in data:

            qid = item["qid"]
            query = item["query"]
            answer = item["answer"]

            response = chain.invoke({"input": query})
            print(f"Query: {query}\nResponse: {response['answer']}")

            similarity = evaluate(response["answer"], answer)
            similarity_scores.append(similarity)

            if similarity > threshold:
                counts += 1
            else:
                print(f"Correct_ANS: {answer}")
            print("")
    elif DATA == "QA_10":
        data = []

        for i in range(10):
            data.append({"qid": i, "query": queries[i], "answer": answers[i]})

        for item in data:

            qid = item["qid"]
            query = item["query"]
            answer = item["answer"]

            response = chain.invoke({"input": query})
            print(f"Query: {query}\nResponse: {response['answer']}")

            if type(answer) == list:
                flag = False
                for ans in answer:
                    similarity = evaluate(response["answer"], ans)
                    similarity_scores.append(similarity)

                    if similarity > threshold:
                        flag = True
                        break
                if not flag:
                    print(f"Correct_ANS: {answer}")
                else:
                    counts += 1
            else:
                similarity = evaluate(response["answer"], answer)
                similarity_scores.append(similarity)

                if similarity > threshold:
                    counts += 1
                else:
                    print(f"Correct_ANS: {answer}")

            print("")

    print(f"Correct numbers: {counts}")
    print(f"Accuracy: {counts/len(data)}")

    sorted_scores = sorted(similarity_scores)

    plt.plot(sorted_scores, range(len(sorted_scores)), "o", markersize=5, alpha=0.7)
    plt.title("Ditrubution of Similarity Scores")
    plt.ylabel("")
    plt.xlabel("Similarity Score")
    plt.xlim(0, max(sorted_scores) + 0.05)
    plt.show()

elif VERSION == "BASIC":
    if DATA == "QA_100":

        with open("QA_100.json", "r", encoding="utf-8") as file:
            data = json.load(file)

        for item in data:

            qid = item["qid"]
            query = item["query"]
            answer = item["answer"]

            response = chain.invoke({"input": query})
            print(f"Query: {query}\nResponse: {response['answer']}")

            if answer.lower() in response["answer"].lower():
                counts += 1
            else:
                print(f"Correct_ANS: {answer}")
            print("")
    elif DATA == "QA_10":
        counts = 0
        for i, query in enumerate(queries):
            # TODO4-3: Run the RAG system
            response = chain.invoke({"input": query})  # Write your code here
            print(f"Query: {query}\nResponse: {response['answer']}")
            # The following lines perform evaluations.
            # if the answer shows up in your response, the response is considered correct.
            correct = False
            if type(answers[i]) == list:
                for answer in answers[i]:
                    answer = answer.split(" ")
                    for ans in answer:
                        if ans.lower() in response["answer"].lower():
                            correct = True
                        else:
                            correct = False
                            break
            else:
                answer = answers[i].split(" ")
                for ans in answer:
                    if ans.lower() in response["answer"].lower():
                        correct = True
                    else:
                        correct = False
                        break
            if correct == False:
                print(f"Uncorrect: {answers[i]}")
            else:
                counts += 1

            print("\n")
    # TODO5: Improve to let the LLM correctly answer the ten questions.
    print(f"Correct numbers: {counts}")
    print(f"Accuracy: {counts/len(queries)}")


# In[ ]:


# import csv

# with open("experiment_2.csv", mode="a") as file:
#     writer = csv.writer(file)
#     try:
#         writer.writerow([EMBED_MODEL, VERSION, DATA, STRICT, counts / len(data)])
#     except:
#         writer.writerow([EMBED_MODEL, VERSION, DATA, STRICT, counts / len(queries)])

