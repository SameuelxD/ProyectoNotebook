# Desarrollado por Jose Samuel Alvarez Silva , 57 3144220093
# Proyecto CRUD con Base de Datos Vectorial (ChromaDB)

Este proyecto implementa un sistema CRUD (Crear, Leer, Actualizar, Eliminar) utilizando una base de datos vectorial (ChromaDB) y embeddings generados por modelos de `Hugging Face`. También incluye ejemplos avanzados de consultas con condiciones y filtros personalizados.

---

## Tabla de Contenidos

1. [Instalación](#instalación)
2. [Configuración de la base de datos vectorial](#configuración-de-la-base-de-datos-vectorial)
3. [Descripción de los métodos CRUD](#descripción-de-los-métodos-crud)
4. [Funcionamiento de los embeddings](#funcionamiento-de-los-embeddings)
5. [Uso de consultas avanzadas](#uso-de-consultas-avanzadas)


---

## Instalación

1. Clona este repositorio:
    ```bash
    git clone https://github.com/SameuelxD/ProyectoNotebook.git
    cd proyectonotebook
    ```

2. Instala las dependencias necesarias:
    ```bash
    pip install os SentenceTransformer chromadb matplotlib.pyplot
    ```

3. Ejecuta el proyecto
    ```bash
    python main.py
    ```

---

## Configuración de la base de datos vectorial

- Se utiliza [ChromaDB](https://www.trychroma.com/) como base de datos vectorial.
- La configuración del cliente de ChromaDB se realiza de forma automática en el script principal:
    ```python
    import chromadb
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="vector_db_demo")
    ```

- Si la colección ya existe, se recupera automáticamente para evitar duplicados.

---

## Descripción de los métodos CRUD

### Crear (Insertar)
- Inserta documentos en la base de datos junto con sus embeddings y metadatos asociados.
    ```python
    create_entry("1", "Texto de ejemplo", {"categoria": "ejemplo"})
    ```

### Leer
- Recupera documentos específicos usando su ID único.
    ```python
    read_entry("1")
    ```

### Actualizar
- Actualiza un documento existente, junto con sus metadatos y embeddings.
    ```python
    update_entry("1", "Nuevo texto", {"categoria": "actualizado"})
    ```

### Eliminar
- Elimina un documento específico de la base de datos por su ID.
    ```python
    delete_entry("1")
    ```

### Consultas avanzadas
- Realiza búsquedas utilizando texto de consulta, embeddings y filtros personalizados.
    ```python
    query_with_filters("Texto de consulta", {"categoria": "ejemplo"})
    ```

---

## Funcionamiento de los embeddings

- Se utiliza el modelo `all-MiniLM-L6-v2` de [Hugging Face](https://huggingface.co/) para generar representaciones vectoriales (embeddings) de textos.
- Los embeddings permiten realizar búsquedas semánticas y comparar textos basados en su significado.

Ejemplo de generación de embeddings:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Texto de ejemplo")
```

---

## Uso de consultas avanzadas

Las consultas en ChromaDB permiten:

1. Buscar documentos similares usando embeddings.
2. Aplicar filtros para limitar los resultados.
3. Personalizar el número de resultados retornados (`top_k`).

Ejemplo:
```python
query_with_filters("¿Qué es ChromaDB?", {"categoria": "base de datos"}, top_k=3)
```

---

