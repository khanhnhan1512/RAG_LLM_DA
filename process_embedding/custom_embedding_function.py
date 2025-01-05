from chromadb import EmbeddingFunction, Embeddings
from langchain_core.documents import Document
import json

METADATA_DESIGN = {
    # metadata for rules
    'rules': {
        'head_rel': 'head_rel',
    },
    
    # metadata for facts
    'facts': {
        'subject': 'subject',
        'relation': 'relation',
        'object': 'object',
        'timestamp': 'timestamp',
    }
}

PAGE_CONTENT_DESIGN = {
    'rules': ['rule'],
    'facts': ['description']
}

def build_documents(df, collection):
    """
    """
    documents = []

    metadata_mapping = METADATA_DESIGN[collection]

    for document_id, row in df.iterrows():
        if collection in PAGE_CONTENT_DESIGN:
            if len(PAGE_CONTENT_DESIGN[collection]) == 1:
                page_content = row[PAGE_CONTENT_DESIGN[collection][0]]
            else:
                page_content = {col: val for col, val in row.to_dict().items() if col in PAGE_CONTENT_DESIGN[collection] and val != ''}
        else:
            page_content = {col: val for col, val in row.to_dict().items() if val != ''}
        
        metadata = {metadata_mapping[col]: row[col] for col in metadata_mapping if col in row}

        document = Document(page_content=json.dumps(page_content, indent=4), metadata=metadata, id=document_id)
        documents.append(document)

    return documents

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, llm_instance):
        self.llm_instance = llm_instance

    def embed_documents(self, pages_content):
        return self.llm_instance.run_embedding(pages_content)
    
    def embed_query(self, query):
        return self.llm_instance.run_embeddings([query])[0]