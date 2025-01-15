import os
from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import matplotlib.pyplot as plt

# Configuración de ChromaDB sin la configuración antigua
chroma_client = chromadb.Client()

# Verificar si la colección ya existe o crearla
collection_name = "vector_db_demo"
if collection_name not in chroma_client.list_collections():
    print(f"Creando colección: {collection_name}")
    collection = chroma_client.create_collection(name=collection_name)
else:
    print(f"Usando colección existente: {collection_name}")
    collection = chroma_client.get_collection(name=collection_name)

# Inicialización del modelo de Hugging Face SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> List[float]:
    """Genera el embedding a partir del texto dado."""
    print(f"Generando embedding para el texto: {text[:50]}...")  # Muestra los primeros 50 caracteres
    embedding = model.encode(text).tolist()
    print(f"Embedding generado (primeros 5 valores): {embedding[:5]}...")  # Muestra los primeros 5 valores
    return embedding

# Crear (Insertar) embeddings en ChromaDB
def create_entry(id: str, text: str, metadata: Dict[str, Any]):
    """Inserta un nuevo documento con su embedding y metadatos en la colección."""
    embedding = generate_embedding(text)
    collection.add(ids=[id], embeddings=[embedding], metadatas=[metadata], documents=[text])
    print(f"Entrada con ID {id} creada correctamente.")

# Leer (Recuperar) entradas desde ChromaDB
def read_entry(id: str) -> Any:
    """Recupera una entrada específica desde la colección por su ID."""
    print(f"Recuperando entrada con ID {id}...")
    result = collection.get(ids=[id])
    if result:
        print(f"Entrada recuperada: {result}")
    else:
        print(f"No se encontró entrada con ID {id}.")
    return result if result else "No entry found with this ID"

# Actualizar una entrada existente en ChromaDB
def update_entry(id: str, new_text: str, new_metadata: Dict[str, Any]):
    """Actualiza el documento de un ID existente con un nuevo texto y metadatos."""
    embedding = generate_embedding(new_text)
    collection.update(ids=[id], embeddings=[embedding], metadatas=[new_metadata], documents=[new_text])
    print(f"Entrada con ID {id} actualizada correctamente.")

# Eliminar una entrada de ChromaDB
def delete_entry(id: str):
    """Elimina una entrada de la colección por su ID."""
    print(f"Eliminando entrada con ID {id}...")
    collection.delete(ids=[id])
    print(f"Entrada con ID {id} eliminada correctamente.")

# Realizar consultas con filtros avanzados
def query_with_filters(query_text: str, filters: Dict[str, Any], top_k: int = 5):
    """Realiza una consulta con el texto de la pregunta y los filtros aplicados."""
    print(f"Realizando consulta con el texto: '{query_text[:50]}...' y filtros: {filters}")
    embedding = generate_embedding(query_text)
    results = collection.query(
        query_embeddings=[embedding], 
        n_results=top_k, 
        where=filters, 
        include=["embeddings", "documents", "metadatas"]  
    )
    if results:
        print(f"Resultados de la consulta: {results}")
    else:
        print("No se encontraron resultados para la consulta.")
    return results if results else "No results found with the given filters"

def demo():
    # Crear entradas
    create_entry("1", "ChromaDB es una base de datos vectorial", {"categoria": "base de datos"})
    create_entry("2", "Hugging Face provee modelos de IA", {"categoria": "IA"})

    # Leer entrada
    print("\n--- Leer entrada ID 1 ---")
    print(read_entry("1"))

    # Actualizar entrada
    print("\n--- Actualizar entrada ID 1 ---")
    update_entry("1", "ChromaDB es una potente base de datos vectorial", {"categoria": "base de datos", "actualizado": True})
    print(read_entry("1"))

    # Consultar con filtros
    print("\n--- Consulta avanzada ---")
    filters = {"categoria": "base de datos"}
    results = query_with_filters("¿Qué es ChromaDB?", filters)
    print(results)

    # Eliminar entrada
    print("\n--- Eliminar entrada ID 1 ---")
    delete_entry("1")
    print(read_entry("1"))

def plot_query_results(results):
    if results:
        documents = [doc for doc in results["documents"]]
        scores = [1 for _ in documents]  # Dummy scores para visualización

        plt.barh(documents, scores, color="blue")
        plt.xlabel("Relevancia")
        plt.title("Resultados de la consulta")
        plt.show()

if __name__ == "__main__":
    demo()

