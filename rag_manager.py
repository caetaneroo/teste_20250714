# Garante compatibilidade com ChromaDB em diferentes ambientes Python
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("pysqlite3 foi ativado para compatibilidade com ChromaDB.")
except ImportError:
    pass

import asyncio
import logging
import os
import shutil
from typing import List, Dict, Any, Optional, Literal, Callable

from dataclasses import dataclass
import chromadb

# --- INÍCIO DA CORREÇÃO ---
# Importa as classes diretamente do pacote 'chonkie', conforme a documentação oficial.
from chonkie import (
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    BaseChunker,
    ChromaHandshake
)
# --- FIM DA CORREÇÃO ---

logger = logging.getLogger(__name__)

# Define os nomes das estratégias disponíveis para o usuário
CHUNKING_STRATEGIES = Literal["semantic", "recursive", "sentence"]

@dataclass
class RetrievalResult:
    """Estrutura para resultados de recuperação."""
    content: str
    similarity: float
    metadata: Dict[str, Any]

class RAGManager:
    """
    Módulo de RAG robusto e encapsulado. Oferece estratégias de chunking
    pré-configuradas e integração otimizada com ChromaDB via ChromaHandshake.
    """
    
    AVAILABLE_STRATEGIES = CHUNKING_STRATEGIES

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

        self.db_client = chromadb.PersistentClient(path=self.vector_store_path)

        self._embedding_func = self._create_embedding_function()
        
        self.handshake = ChromaHandshake(
            embedding_function=self._embedding_func,
            collection_name="default"
        )

        self.chunkers: Dict[CHUNKING_STRATEGIES, BaseChunker] = {
            "semantic": SemanticChunker(
                embedding_function=self._embedding_func,
                similarity_cutoff=0.6 
            ),
            "recursive": RecursiveChunker(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=1000,
                chunk_overlap=100,
            ),
            "sentence": SentenceChunker()
        }
        
        logger.info(
            f"RAGManager inicializado. Estratégias disponíveis: {', '.join(self.chunkers.keys())}"
        )

    def _create_embedding_function(self) -> Callable[[List[str]], List[List[float]]]:
        """Cria uma função de embedding síncrona, compatível com Chonkie e ChromaDB."""
        async def embed(texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            response = await self.ai_processor.process_embedding_batch(texts)
            return [
                res['content'] if res.get('success') else []
                for res in response['results']
            ]
        
        def embedding_function(texts: List[str]) -> List[List[float]]:
            return asyncio.run(embed(texts))
        
        return embedding_function

    async def ingest_documents(
        self,
        collection_name: str,
        strategy: CHUNKING_STRATEGIES = "semantic",
        relative_path: str = '',
        force_update: bool = False
    ) -> int:
        """
        Ingere documentos usando uma estratégia de chunking pré-configurada.
        - collection_name: Nome da coleção no ChromaDB.
        - strategy: Estratégia de chunking a ser usada ('semantic', 'recursive', 'sentence').
        - relative_path: Subpasta dentro de 'documents/'.
        - force_update: Força re-ingestão de todos os arquivos.
        Retorna o número de arquivos processados.
        """
        ingest_dir = os.path.join(self.documents_dir, relative_path)
        if not os.path.exists(ingest_dir):
            raise FileNotFoundError(f"Diretório '{ingest_dir}' não encontrado.")
            
        if force_update:
            self.clear_vector_store(collection_name)

        collection = self.db_client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        
        chonker = self.chunkers.get(strategy)
        if not chonker:
            raise ValueError(f"Estratégia de chunking '{strategy}' inválida. Disponíveis: {list(self.chunkers.keys())}")

        self.handshake.collection_name = collection_name

        files_to_process = [f for f in os.listdir(ingest_dir) if os.path.isfile(os.path.join(ingest_dir, f))]
        
        for file_name in files_to_process:
            if not force_update and self._is_file_ingested(collection, file_name):
                logger.info(f"Arquivo '{file_name}' já ingerido. Pulando.")
                continue

            with open(os.path.join(ingest_dir, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            logger.info(f"Processando '{file_name}' com a estratégia '{strategy}'...")
            
            # Roda a função síncrona 'run' do handshake em um thread separado para não bloquear o loop async.
            await asyncio.to_thread(
                self.handshake.run,
                text=content,
                chonker=chonker,
                collection=collection,
                metadata={'file': file_name}
            )

        logger.info(f"Ingestão concluída para '{collection_name}': {len(files_to_process)} arquivos avaliados.")
        return len(files_to_process)

    def _is_file_ingested(self, collection: chromadb.Collection, file_name: str) -> bool:
        result = collection.get(where={"file": file_name}, limit=1)
        return bool(result['ids'])

    async def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[RetrievalResult]:
        """Recupera chunks semelhantes à query de uma coleção no ChromaDB."""
        try:
            collection = self.db_client.get_collection(name=collection_name)
        except ValueError:
            raise ValueError(f"Coleção '{collection_name}' não encontrada.")
        
        query_embedding_list = await asyncio.to_thread(self._embedding_func, [query])
        query_embedding = query_embedding_list[0]
        if not query_embedding:
            raise RuntimeError("Falha ao embeddar a query de busca.")

        # Executa a busca síncrona do ChromaDB em um thread separado.
        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieval_results = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i]
                retrieval_results.append(
                    RetrievalResult(
                        content=results['documents'][0][i],
                        similarity=1 - distance,
                        metadata=results['metadatas'][0][i]
                    )
                )
        return retrieval_results

    def inspect_collection(self, collection_name: str, limit: int = 5) -> Optional[Dict[str, Any]]:
        """Inspeciona o conteúdo de uma coleção no ChromaDB."""
        try:
            collection = self.db_client.get_collection(name=collection_name)
            count = collection.count()
            if count == 0:
                print(f"Coleção '{collection_name}' existe mas está vazia.")
                return {'name': collection_name, 'count': 0, 'sample': []}
                
            sample = collection.get(limit=limit, include=["metadatas", "documents"])
            
            print(f"--- Inspeção da Coleção: '{collection_name}' ---")
            print(f"Total de itens: {count}")
            for i, doc_id in enumerate(sample['ids']):
                print(f"  - ID: {doc_id} | Metadata: {sample['metadatas'][i]}")
                print(f"    Documento: \"{sample['documents'][i][:120].strip()}...\"")
            print("-" * 43)
            return {'name': collection_name, 'count': count, 'sample': sample}
        except ValueError:
            print(f"Coleção '{collection_name}' não encontrada.")
            return None

    def enrich_prompts(self, base_prompts: List[str], retrieved: List[RetrievalResult], context_template: str = "Use o seguinte contexto para responder à pergunta:\n\n---\n{context}\n---\n") -> List[str]:
        if not retrieved:
            return base_prompts
        combined_context = "\n\n".join([res.content for res in retrieved])
        return [context_template.format(context=combined_context) + prompt for prompt in base_prompts]

    def clear_vector_store(self, collection_name: Optional[str] = None):
        """Limpa uma coleção específica ou todo o armazenamento vetorial."""
        if collection_name:
            try:
                self.db_client.delete_collection(name=collection_name)
                logger.info(f"Coleção '{collection_name}' removida.")
            except ValueError:
                pass
        else:
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.db_client = chromadb.PersistentClient(path=self.vector_store_path)
            logger.info("Todo o armazenamento vetorial foi limpo e recriado.")
