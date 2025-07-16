import asyncio
import json
import logging
import os
import shutil
from typing import List, Dict, Any, Optional, Literal

from dataclasses import dataclass
# Bibliotecas de terceiros para robustez
import chromadb
from markdown_chunker import MarkdownChunkingStrategy

logger = logging.getLogger(__name__)

# Constantes padrão
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 1024 # Tamanho padrão para chunking

# Tipos para chunking
ChunkingStrategy = Literal["simple", "markdown"]

@dataclass
class RetrievalResult:
    """Estrutura para resultados de recuperação."""
    content: str
    similarity: float
    metadata: Dict[str, Any]

class RAGManager:
    """
    Módulo para Retrieval-Augmented Generation (RAG) robusto, integrado ao AIProcessor.
    Utiliza ChromaDB para armazenamento vetorial e estratégias de chunking flexíveis.
    """

    def __init__(self, ai_processor: 'AIProcessor', project_dir: str):
        """
        Inicializa o RAGManager.
        - ai_processor: Instância do AIProcessor para embeddings.
        - project_dir: Caminho para a pasta do projeto.
        """
        self.ai_processor = ai_processor
        self.project_dir = os.path.abspath(project_dir)
        self.documents_dir = os.path.join(self.project_dir, 'documents')
        self.vector_store_path = os.path.join(self.project_dir, 'vector_store')

        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)

        # 1. ARMAZENAMENTO VETORIAL ROBUSTO: ChromaDB
        # Usa o cliente persistente para salvar os dados no disco.
        self.db_client = chromadb.PersistentClient(path=self.vector_store_path)
        
        # Inicializa a estratégia de chunking de markdown
        self.markdown_chunker = MarkdownChunkingStrategy(
            min_chunk_len=256, 
            soft_max_len=DEFAULT_CHUNK_SIZE, 
            hard_max_len=DEFAULT_CHUNK_SIZE * 2
        )

        logger.info(f"RAGManager inicializado. Armazenamento vetorial em '{self.vector_store_path}'.")

    async def ingest_documents(
        self,
        collection_name: str,
        relative_path: str = '',
        force_update: bool = False,
        chunk_strategy: ChunkingStrategy = "simple",
        chunk_size: int = DEFAULT_CHUNK_SIZE
    ) -> int:
        """
        Ingestão de documentos: lê arquivos, chunkifica, embedda e armazena no ChromaDB.
        - collection_name: Nome da coleção no ChromaDB para agrupar os documentos.
        - relative_path: Subpasta dentro de 'documents/' (opcional).
        - force_update: Força re-ingestão de todos os arquivos.
        - chunk_strategy: 'simple' (baseado em frases) ou 'markdown' (estruturado).
        - chunk_size: Tamanho alvo de cada chunk em caracteres (usado no modo 'simple').
        Retorna o número de arquivos processados.
        """
        ingest_dir = os.path.join(self.documents_dir, relative_path)
        if not os.path.exists(ingest_dir):
            raise FileNotFoundError(f"Diretório '{ingest_dir}' não encontrado.")
            
        if force_update:
            self.clear_vector_store(collection_name)

        collection = self.db_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Configura para usar similaridade de cosseno
        )

        files_to_process = [f for f in os.listdir(ingest_dir) if os.path.isfile(os.path.join(ingest_dir, f))]
        
        for file_name in files_to_process:
            if not force_update and self._is_file_ingested(collection, file_name):
                logger.info(f"Arquivo '{file_name}' já ingerido na coleção '{collection_name}'. Pulando.")
                continue

            with open(os.path.join(ingest_dir, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 2. ESTRATÉGIAS DE CHUNKING FLEXÍVEIS
            if chunk_strategy == "markdown":
                chunks = self.markdown_chunker.chunk_markdown(content)
            else: # 'simple'
                chunks = self._chunk_text_simple(content, chunk_size)

            if not chunks:
                logger.warning(f"Nenhum chunk gerado para o arquivo '{file_name}'. Pulando.")
                continue
                
            logger.info(f"Processando {len(chunks)} chunks de '{file_name}' para a coleção '{collection_name}'...")

            emb_response = await self.ai_processor.process_embedding_batch(
                chunks, batch_id=f"ingest_{collection_name}_{file_name}"
            )
            
            embeddings = [res['content'] for res in emb_response['results'] if res['success']]
            successful_chunks = [chunks[i] for i, res in enumerate(emb_response['results']) if res['success']]

            if not successful_chunks:
                logger.error(f"Falha ao gerar embeddings para todos os chunks de '{file_name}'.")
                continue

            ids = [f"{file_name}_chunk_{i}" for i in range(len(successful_chunks))]
            metadatas = [{'file': file_name, 'chunk_id': i} for i in range(len(successful_chunks))]
            
            collection.add(
                documents=successful_chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

        logger.info(f"Ingestão concluída para a coleção '{collection_name}': {len(files_to_process)} arquivos avaliados.")
        return len(files_to_process)

    def _is_file_ingested(self, collection: chromadb.Collection, file_name: str) -> bool:
        """Verifica se algum documento de um arquivo já existe na coleção."""
        result = collection.get(where={"file": file_name}, limit=1)
        return bool(result['ids'])

    def _chunk_text_simple(self, text: str, chunk_size: int) -> List[str]:
        """Chunking simples: divide texto em pedaços, respeitando frases."""
        chunks, current_chunk = [], ""
        for sentence in text.split('. '):
            if len(current_chunk) + len(sentence) + 2 > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += sentence + '. '
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    async def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> List[RetrievalResult]:
        """
        Recupera chunks semelhantes à query de uma coleção no ChromaDB.
        """
        try:
            collection = self.db_client.get_collection(name=collection_name)
        except ValueError:
            raise ValueError(f"Coleção '{collection_name}' não encontrada. Execute ingest_documents primeiro.")
        
        query_emb_result = await self.ai_processor.process_embedding_batch([query], batch_id="retrieve_query")
        if not query_emb_result['results'][0]['success']:
            raise RuntimeError("Falha ao embeddar a query de busca.")
        query_embedding = query_emb_result['results'][0]['content']

        # 3. BUSCA DE SIMILARIDADE OTIMIZADA COM CHROMADB
        # ChromaDB retorna 'distances'. Para cosseno, distância = 1 - similaridade.
        # Portanto, filtramos onde a distância é MENOR que (1 - threshold).
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"$and": [{"distance": {"$lte": 1 - similarity_threshold}}]} # Filtro customizado de metadados não é padrão, filtro de distância sim.
        )
        
        retrieval_results = []
        # O resultado de query é uma lista, uma para cada query_embedding. Pegamos a primeira.
        if results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i]
                similarity = 1 - distance
                if similarity >= similarity_threshold:
                    retrieval_results.append(
                        RetrievalResult(
                            content=results['documents'][0][i],
                            similarity=similarity,
                            metadata=results['metadatas'][0][i]
                        )
                    )
        
        # A ordenação já é feita pelo ChromaDB.
        return retrieval_results

    def enrich_prompts(self, base_prompts: List[str], retrieved: List[RetrievalResult], context_template: str = "Contexto relevante: {context}\n") -> List[str]:
        """Enriquece uma lista de prompts com contextos recuperados."""
        if not retrieved:
            return base_prompts
            
        contexts = [context_template.format(context=res.content) for res in retrieved]
        combined_context = "\n".join(contexts)
        
        return [combined_context + prompt for prompt in base_prompts]

    def clear_vector_store(self, collection_name: Optional[str] = None):
        """
        Limpa uma coleção específica ou todo o armazenamento vetorial.
        - collection_name: Se fornecido, apaga apenas essa coleção. Senão, apaga TUDO.
        """
        if collection_name:
            try:
                self.db_client.delete_collection(name=collection_name)
                logger.info(f"Coleção '{collection_name}' removida do ChromaDB.")
            except ValueError:
                logger.warning(f"Tentativa de apagar coleção '{collection_name}' que não existe.")
        else:
            # Apaga todo o diretório do ChromaDB para uma limpeza completa.
            shutil.rmtree(self.vector_store_path)
            os.makedirs(self.vector_store_path, exist_ok=True)
            # Recria o cliente para operar sobre o diretório limpo.
            self.db_client = chromadb.PersistentClient(path=self.vector_store_path)
            logger.info("Todo o armazenamento vetorial foi limpo e recriado.")
