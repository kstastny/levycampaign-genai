from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer


# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openai_organization = os.getenv("OPENAI_ORGANIZATION")
openai_project_id = os.getenv("OPENAI_PROJECT_ID")

splitter_name = os.getenv("SPLITTER")

embeddingModelName = "paraphrase-MiniLM-L6-v2"
embeddingModel = SentenceTransformer(embeddingModelName)

collection_name = f"levycampaign_{splitter_name.lower()}_{embeddingModelName.replace('-', '')}"

print(f"Qdrant Collection name: {collection_name}")

llmmodel = "gpt-4o-mini"