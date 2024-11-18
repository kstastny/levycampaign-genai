# Levy&Campaign Assistant

*This is a toy/proof of concept command line project for creating an assistant that will help with rules questions and strategy for Levy&Campaign games*

## Current Status

PoC of Almoravid in progress, because of PoC status, contains almost no configuration


## How to run locally

### Prerequisities

- OpenAI API access
- Python 3.11.8+
- Docker for running QDrant
- rules of Almoravid saved as "data/Almoravid+Rules+of+Play+-+LIVING+RULES+(1).pdf" 
    - available at https://www.gmtgames.com/p-861-almoravid-reconquista-and-riposte-in-spain-1085-1086.aspx

### Environment Preparation

1. Install QDrant database, see e.g. https://qdrant.tech/documentation/guides/installation/#docker

    ```bash
    docker pull qdrant/qdrant
    ```

1. Run QDrant

    ```bash
    docker run -p 6333:6333 \
        -v $(pwd)/qdrant_data:/qdrant/storage \
        qdrant/qdrant
    ```

1. Setup .env file

    ```
    OPENAI_API_KEY=<<YOUR OPENAI API KEY>>
    OPENAI_ORGANIZATION=<<YOUR OPENAI ORG KEY>>
    OPENAI_PROJECT_ID=<<YOUR OPENAI PROJECT KEY KEY>>

    SPLITTER=SentenceSplitter
    #SPLITTER=SemanticSplitter
    ```

1. If required, prepare Python virtual environment and activate it

    ```bash
    python -m venv testenv
    ```
    ```bash
    testenv\Scripts\activate
    ```    

1. Install prerequisities

    ```bash
    pip install -qU llama-index-embeddings-openai llama-index-readers-file python-dotenv qdrant_client sentence-transformers openai rouge-score
    ```


### Running the app

0. Ignore the Python notebooks, there is just garbage there and they have to be cleaned

1. Prepare data in the Search database

    ```bash
    python prepare_data.py
    ```

1. Run the Assistant

    ```bash
    python chat.py
    ```    
