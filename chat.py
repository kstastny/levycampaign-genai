# venv\Scripts\activate

from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from dotenv import load_dotenv
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datetime import datetime

# https://thepythoncode.com/article/calculate-rouge-score-in-python
from rouge_score import rouge_scorer
import numpy as np



from core import collection_name, openai_key, openai_organization, openai_project_id
from core import embeddingModel, llmmodel


print("Initializing neurons...")

openaiclient = OpenAI(
  organization=openai_organization,
  project=openai_project_id,
  api_key=openai_key
)


qdrant = QdrantClient(path="./qdrant_data")


def debug_hit (score, payload, filename):
    with open(filename, "w") as file:
        file.write("Score: " + str(score) + "\n")
        file.write("Payload: " + str(payload) + "\n")

def retrieveDocuments (query):
    hits = qdrant.search(
        collection_name=collection_name,
        query_vector=embeddingModel.encode(query).tolist(),
        limit=10,
    )

    sorted_documents = sorted(hits, key=lambda hit: hit.score, reverse=True)

    for index, hit in enumerate(sorted_documents):
        debug_hit(hit.score, hit.payload, f"debug/hit_{index}.txt")

    return sorted_documents

documents_to_retrieve = 3

def retriever(query):
    documents = retrieveDocuments(query)
    #context = documents[0].payload["source"]

    context = "\n\n---\n\n".join([doc.payload["source"] for doc in documents[:documents_to_retrieve]])

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    with open(f"debug/qa/{formatted_date}.txt", "w", encoding="utf-8") as file:
        file.write("Query: " + user_query + "\n")
        file.write("\n")
        file.write("--------------------")
        file.write("\n")
        file.write("Context: " + context + "\n")    

    return context


def generateAnswerDirect (context, query):
    answer = openaiclient.chat.completions.create(
        model = llmmodel,
        messages = [
            {"role": "system",
                "content": 
                  """You are an expert player of Levy & Campaign games, particularly Almoravid.
                    You are helpful and always willing to explain the rules and strategies of the game.
                    Please explain the rules based on the context only. If you don't know the answer, say I don't know."""
            },
            {"role": "system", "content": context},
            {"role": "user", "content": query}
        ]
    )
    return answer.choices[0].message.content


def askllmDirect(query):
    context = retriever(query)
    answer = generateAnswerDirect(context, query)
    return answer





## Langchain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI    

prompt = PromptTemplate.from_template("""You are an expert player of Levy & Campaign games, particularly Almoravid. You are helpful and always willing to explain the rules and strategies of the game.
                    Please explain the rules based on the context only.
                    If you don't know the answer, say I don't know. \nQuestion: {question} \nContext: {context} \nAnswer:""")

llm = ChatOpenAI(model=llmmodel)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def askllmLangchain (query):
    #return rag_chain.invoke(question=query) # returns extra information
    return rag_chain.invoke(query)

def askllmStream (query):
    return rag_chain.stream(query)

# choose how the answer will be generated
askllm = askllmLangchain

def test():
    print("Running self-evaluation test...")

    test_queries_answers = [
        ("What is the objective of the game?","The objective of the game is to score the most points by the end of the given scenario."),
        ("When does the game take place?", "The game takes place during mediaval reconquista in Iberian peninsula, specifically in the years 1085-1086"),
        ("In what historical period does the game take place?", "The game takes place during mediaval reconquista in Iberian peninsula, specifically in the years 1085-1086"),
        ("What do the opponents represent?", "The opponents represent the muslim Taifas (green color) and the Christian forces (yellow) led by Alphonso VI of Leon and Castile"),
        ("Tell me what this game is about", 
         """Almoravid is a historical boardgame from the Levy&Campaign series that represents a period of conquest in the second half of 11th century.
         It shows the fight of king Alphonso against the muslim Taifas and the Almoravid forces."""),
        ("What are the different phases of the turn?", 
         """During a Campaign, both players stack several Command cards into a facedown Campaign Plan. They then alternate,
         starting with the Christian player, in revealing and executing the Command cards.
         Each command allows the player to Pass, or execute an action like Move or Supply.
         If necessary, the player than has to Feed and Pay their Forces, or Disband a Lord."""),
        ("When does Taifa change to Reconquista?", 
         """When all of Taifa's Cities are conquered by a Christian player, the Taifa's status changes to Reconquista."""),
        ("Can Muslims and Christians attack strongholds in neutral/Parias Taifas?", 
         """Generally, neither Muslim nor Christian player can attack strongholds in neutral Taifas. However, if the locale is Conquered, has Jihad or
         Seat markers of an opponent, then it may be attacked."""),
        ("Can a Muslim attack a Neutral locale?", 
         """No, a Muslim cannot attack a Neutral locale. Strongholds with a Neutral status cannot be besieged"""),
        ("What does it mean when the Lord is Laden?", 
         """Laden Lord carries more Provender than they can normally manage. Lord is Laden when carrying Loot, using Cart to move
         over a Pass or when a Cart or Mule carries two Provender."""),           
         
    ]

    now = datetime.now()
    formatted_date = now.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    os.makedirs(f"debug/test", exist_ok=True)

    with open(f"debug/test/{formatted_date}.txt", "w", encoding="utf-8") as file:
        for query, expected_answer in test_queries_answers:
            answer = askllm(query)
            scores = scorer.score(expected_answer, answer)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)            

            file.write("Query: " + query + "\n")
            file.write(f"ROUGE-1 F1 Score: {scores['rouge1'].fmeasure}\n")
            file.write(f"ROUGE-2 F1 Score: {scores['rouge2'].fmeasure}\n")
            file.write(f"ROUGE-L F1 Score: {scores['rougeL'].fmeasure}\n")
            file.write("Answer: " + answer + "\n")
            #file.write("Context: " + context + "\n")
            file.write("\n")
            file.write("--------------------")
            file.write("\n")


        avg_rouge1 = np.mean(rouge1_scores)
        avg_rouge2 = np.mean(rouge2_scores)
        avg_rougeL = np.mean(rougeL_scores)

        print(f"Average ROUGE-1 F1 Score: {avg_rouge1:.4f}")
        print(f"Average ROUGE-2 F1 Score: {avg_rouge2:.4f}")
        print(f"Average ROUGE-L F1 Score: {avg_rougeL:.4f}")
        file.write("====================================\n")
        file.write(f"Average ROUGE-1 F1 Score: {avg_rouge1:.4f}\n")
        file.write(f"Average ROUGE-2 F1 Score: {avg_rouge2:.4f}\n")
        file.write(f"Average ROUGE-L F1 Score: {avg_rougeL:.4f}\n")    



print(" Hello, I am a chatbot that will help you with Almoravid. How can I help you? Write 'exit' to leave the program, 'test' to run self-evaluation test.")
os.makedirs(f"debug/qa", exist_ok=True)

user_query = ""
while user_query != "exit":

    if user_query == "test":
        test()
    elif user_query != "":
        #print ("Echo Chamber: ", user_query)
        #context, answer = askllm(user_query)
        #print("Answer: ", answer)
        
        answer = askllmStream(user_query)
        for chunk in answer:
            print(chunk, end="", flush=True)
        print("\n")

        
        

    user_query = input(" > ")


qdrant.close()    