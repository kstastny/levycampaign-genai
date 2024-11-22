import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from core import embeddingModel, collection_name, splitter_name

# rules available at https://www.gmtgames.com/p-861-almoravid-reconquista-and-riposte-in-spain-1085-1086.aspx

game_docs = [
    "data/Almoravid+Rules+of+Play+-+LIVING+RULES+(1).pdf"
]

# load documents
documents = SimpleDirectoryReader(input_files=game_docs).load_data()

text = "\n".join([doc.text for doc in documents])

print(f" Processing text of length {len(text)}")



# Initialize the SentenceSplitter
split_text = None
if splitter_name == "SentenceSplitter":
    splitter = SentenceSplitter()
    split_text = splitter.split_text
elif splitter_name == "SemanticSplitter":
    from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser)
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import Document

    embed_model = OpenAIEmbedding()
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
        #,sentence_splitter=SentenceSplitter(chunk_size=1024)
    )

    def split_text (text):
        #TODO work with related nodes? use VectorStore directly https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/#setup-query-engine
        nodes = splitter.get_nodes_from_documents([Document(text=text, id="1")], show_progress=True)
        return [ n.get_content() for n in nodes]

else:
    raise ValueError(f"Unknown splitter name: {splitter_name}")


text_chunks = split_text(text)

print(f"Text split into {len(text_chunks)} chunks")

os.makedirs(f"debug/{splitter_name}", exist_ok=True)
for index, sentence in enumerate(text_chunks):
    filename = f"debug/{splitter_name}/{index:03}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(sentence)


## generate embeddings
vectors = [embeddingModel.encode(sentence) for sentence in text_chunks]
vector_size = len(vectors[0])


#store data
from qdrant_client import models, QdrantClient
qdrant = QdrantClient(path="./qdrant_data")

qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size, # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)

qdrant.upsert(
    collection_name=collection_name,
    points=[
        models.PointStruct(
            id=idx, 
            vector=vectors[idx].tolist(),
            # place where dict with the vector source to be added or id/reference for the source
            payload={"source": doc},
        )
        for idx, doc in enumerate(text_chunks)
    ],
)



print("Processing done")

print("Current Qdrant collections: ")
for collection in qdrant.get_collections().collections:
    print(f"\t{collection}")

# TODO run sample queries, allow user to run queries and return documents