import os
import glob
import json
import numpy as np
from typing import List, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, cache_dir: str = "cache"):
        print(f"\n{'='*50}\nInitializing RAG System...")
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.documents = []
        self.embeddings = None
        self.embedding_model = SentenceTransformer("nomic-ai/modernbert-embed-base")
        print(f"✓ Loaded embedding model: nomic-ai/modernbert-embed-base")
        
        # Setup cache directory
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"✓ Created cache directory at {cache_dir}")
        else:
            print(f"✓ Using existing cache directory at {cache_dir}")
        print(f"{'='*50}\n")
        
    def load_documents(self, directory: str = "data"):
        """Load all text documents from the specified directory."""
        print(f"\n{'='*50}\nLoading documents from {directory}...")
        file_paths = glob.glob(os.path.join(directory, "*.txt"))
        
        if not file_paths:
            raise ValueError(f"No .txt files found in {directory}")
        
        total_chars = 0
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                total_chars += len(content)
                self.documents.append({
                    'content': content,
                    'source': os.path.basename(file_path)
                })
            print(f"✓ Loaded {os.path.basename(file_path)} ({len(content)} chars)")
        
        print(f"\nSummary:")
        print(f"✓ Total documents loaded: {len(self.documents)}")
        print(f"✓ Total characters: {total_chars:,}")
        print(f"{'='*50}\n")

    def save_embeddings(self):
        """Save embeddings and document metadata to cache."""
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Run create_embeddings() first.")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save embeddings
        embeddings_file = os.path.join(self.cache_dir, f"embeddings_{timestamp}.npy")
        np.save(embeddings_file, self.embeddings)
        
        # Save document metadata
        metadata = [{
            'source': doc['source'],
            'content_length': len(doc['content'])
        } for doc in self.documents]
        
        metadata_file = os.path.join(self.cache_dir, f"metadata_{timestamp}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
            
        print(f"✓ Saved embeddings to {embeddings_file}")
        print(f"✓ Saved metadata to {metadata_file}")
        return embeddings_file, metadata_file

    def load_cached_embeddings(self, embeddings_file: str, metadata_file: str):
        """Load embeddings and verify against current documents."""
        print(f"\n{'='*50}\nLoading cached embeddings...")
        
        # Load embeddings
        self.embeddings = np.load(embeddings_file)
        print(f"✓ Loaded embeddings shape: {self.embeddings.shape}")
        
        # Load and verify metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
            
        if len(metadata) != len(self.documents):
            raise ValueError(f"Cached metadata length ({len(metadata)}) doesn't match current documents ({len(self.documents)})")
        
        # Verify each document matches metadata
        for doc, meta in zip(self.documents, metadata):
            if doc['source'] != meta['source'] or len(doc['content']) != meta['content_length']:
                raise ValueError(f"Document mismatch for {doc['source']}")
                
        print(f"✓ Verified metadata matches current documents")
        print(f"{'='*50}\n")

    def create_embeddings(self):
        """Create embeddings for all loaded documents using SentenceTransformer."""
        print(f"\n{'='*50}\nCreating embeddings...")
        
        if not self.documents:
            raise ValueError("No documents loaded. Run load_documents() first.")
            
        texts = [doc['content'] for doc in self.documents]
        print(f"✓ Processing {len(texts)} documents")
        
        start_time = datetime.now()
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"\nSummary:")
        print(f"✓ Created embeddings of shape: {self.embeddings.shape}")
        print(f"✓ Processing time: {duration:.2f} seconds")
        print(f"✓ Average time per document: {duration/len(texts):.2f} seconds")
        print(f"{'='*50}\n")

    def get_relevant_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve the most relevant documents for a query."""
        print(f"\nFinding relevant documents for query: '{query}'")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        print(f"✓ Created query embedding of shape: {query_embedding.shape}")
        
        # Calculate similarities using dot product
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_docs = []
        print(f"\nTop {top_k} most relevant documents:")
        for idx in top_indices:
            doc = {
                'content': self.documents[idx]['content'],
                'source': self.documents[idx]['source'],
                'similarity': float(similarities[idx])
            }
            relevant_docs.append(doc)
            print(f"✓ {doc['source']} (similarity: {doc['similarity']:.3f})")
            #print preview of the content
            print(f"Preview: {doc['content'][:4000]}...")
            print(f"{'='*50}\n")
        
        return relevant_docs

    def query(self, user_query: str) -> str:
        """Process a user query and return a response."""
        print(f"\n{'='*50}")
        print(f"Processing query: '{user_query}'")
        
        # Get relevant documents
        relevant_docs = self.get_relevant_documents(user_query)
        
        # Construct the prompt
        context = "\n\n".join([
            f"Document {i+1} (Similarity: {doc['similarity']:.3f}):\n{doc['content']}" 
            for i, doc in enumerate(relevant_docs)
        ])
        
        prompt = f"""Based on the following documents about tenant rights in Ontario, please answer this question:

Question: {user_query}

Relevant Documents:
{context}

Please provide a clear and accurate answer based only on the information provided in these documents. If the information needed is not in the documents, please state that."""

        print("\nGenerating response using GPT-4...")
        # Get response from OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information about tenant rights in Ontario based on the provided documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        print(f"✓ Response generated ({len(answer)} chars)")
        print(f"{'='*50}\n")
        return answer

def create_and_cache_embeddings(data_dir: str = "data", cache_dir: str = "cache"):
    """Utility function to create and cache embeddings in one go."""
    rag = RAGSystem(cache_dir=cache_dir)
    rag.load_documents(data_dir)
    rag.create_embeddings()
    return rag.save_embeddings()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG System for Ontario Tenant Rights')
    parser.add_argument('--create-cache', action='store_true', 
                      help='Create and cache embeddings only')
    parser.add_argument('--use-cache', nargs=2, metavar=('EMBEDDINGS', 'METADATA'),
                      help='Use cached embeddings and metadata files')
    
    args = parser.parse_args()
    
    if args.create_cache:
        print("Creating and caching embeddings...")
        embeddings_file, metadata_file = create_and_cache_embeddings()
        print(f"\nEmbeddings cached! To use them later, run:")
        print(f"python rag_system.py --use-cache {embeddings_file} {metadata_file}")
        exit(0)
    
    # Initialize the system
    rag = RAGSystem()
    
    # Load documents
    rag.load_documents()
    
    # Either create new embeddings or load cached ones
    if args.use_cache:
        rag.load_cached_embeddings(args.use_cache[0], args.use_cache[1])
    else:
        rag.create_embeddings()
    
    print("\nRAG System is ready! Enter your questions about tenant rights in Ontario.")
    print("Type 'quit' to exit.")
    
    # Example usage
    while True:
        try:
            query = input("\nYour question: ")
            if query.lower() == 'quit':
                break
                
            response = rag.query(query)
            print("\nResponse:", response)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again or type 'quit' to exit.") 