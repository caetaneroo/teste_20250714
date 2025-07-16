# Garante compatibilidade com ChromaDB, de forma silenciosa e transparente.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import asyncio
import logging
import os
import shutil
from typing import List, Dict, Any, Optional, Literal, Union

from dataclasses import dataclass
import chromadb

# Importa as classes da biblioteca 'chonkie' diretamente, conforme a documentação oficial.
from chonkie import (
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    TokenChunker  # Alternativa mais estável ao SemanticChunker
)

logger = logging.getLogger(__name__)

# Constantes de configuração padrão do módulo
DEFAULT_TOP_K = 5
CHUNKING_STRATEGIES = Literal["semantic", "recursive", "sentence", "token"]

@dataclass
class RetrievalResult:
    """Estrutura padronizada para resultados de recuperação."""
    content: str
    similarity: float
    metadata: Dict[str, Any]

class RAGManager:
    """
    Módulo de RAG robusto e encapsulado. Oferece estratégias de chunking
    pré-configuradas e integração otimizada com ChromaDB.
    """
    
    def __init__(self, ai_processor: 'AIProcessor', project_dir: str):
        """
        Inicializa o RAGManager.
        - ai_processor: Instância do AIProcessor, usada para gerar embeddings.
        - project_dir: Caminho para a pasta do projeto onde os dados serão armazenados.
        """
        self.ai_processor = ai_processor
        self.project_dir = os.path.abspath(project_dir)
        self.documents_dir = os.path.join(self.project_dir, 'documents')
        self.vector_store_path = os.path.join(self.project_dir, 'vector_store')

        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)

        self.db_client = chromadb.PersistentClient(path=self.vector_store_path)

        # Configuração correta dos chunkers
        self.chunkers: Dict[CHUNKING_STRATEGIES, Union[SentenceChunker, RecursiveChunker, SemanticChunker, TokenChunker]] = {
            "semantic": SemanticChunker(
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                threshold=0.6,
                chunk_size=512,
                min_sentences=1
            ),
            "recursive": RecursiveChunker(
                chunk_size=1000,
                tokenizer="gpt2"
            ),
            "sentence": SentenceChunker(
                chunk_size=1000,
                tokenizer="gpt2"
            ),
            "token": TokenChunker(
                chunk_size=1000,
                chunk_overlap=100,
                tokenizer="gpt2"
            )
        }
        
        logger.info(
            f"RAGManager inicializado. Estratégias disponíveis: {', '.join(self.chunkers.keys())}"
        )

    async def ingest_documents(
        self,
        collection_name: str,
        strategy: CHUNKING_STRATEGIES = "token",  # Mudança para estratégia mais estável
        relative_path: str = '',
        force_update: bool = False
    ) -> int:
        """
        Ingere documentos usando uma orquestração manual e robusta.
        - collection_name: Nome da coleção no ChromaDB.
        - strategy: Estratégia de chunking a ser usada.
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
        
        chunker = self.chunkers.get(strategy)
        if not chunker:
            raise ValueError(f"Estratégia '{strategy}' inválida. Disponíveis: {list(self.chunkers.keys())}")

        files_to_process = [f for f in os.listdir(ingest_dir) if os.path.isfile(os.path.join(ingest_dir, f))]
        processed_files = 0
        
        for file_name in files_to_process:
            if not force_update and await self._is_file_ingested(collection, file_name):
                logger.info(f"Arquivo '{file_name}' já ingerido. Pulando.")
                continue

            try:
                with open(os.path.join(ingest_dir, file_name), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                logger.info(f"Processando '{file_name}' com a estratégia '{strategy}'...")

                # 1. Chunking (operação síncrona em thread separada) - CORREÇÃO CRÍTICA
                try:
                    chunk_objects = await asyncio.to_thread(chunker.chunk, content)
                    chunks = [chunk.text for chunk in chunk_objects]  # Extrair texto dos objetos
                except Exception as e:
                    logger.error(f"Erro no chunking de '{file_name}': {e}")
                    continue
                    
                if not chunks:
                    logger.warning(f"Nenhum chunk gerado para o arquivo '{file_name}'.")
                    continue

                # 2. Embedding (operação assíncrona nativa)
                embedding_results = await self.ai_processor.process_embedding_batch(chunks)

                # 3. Preparação dos Dados
                docs_to_add, embeddings_to_add, metadatas_to_add, ids_to_add = [], [], [], []
                for i, result in enumerate(embedding_results['results']):
                    if result.get('success'):
                        docs_to_add.append(chunks[i])
                        embeddings_to_add.append(result['content'])
                        metadatas_to_add.append({'file': file_name, 'chunk_id': i})
                        ids_to_add.append(f"{file_name}_{i}")

                if not docs_to_add:
                    logger.error(f"Falha ao gerar embeddings para todos os chunks de '{file_name}'.")
                    continue

                # 4. Adição ao ChromaDB (operação síncrona em thread separada)
                try:
                    await asyncio.to_thread(
                        collection.add,
                        embeddings=embeddings_to_add,
                        documents=docs_to_add,
                        metadatas=metadatas_to_add,
                        ids=ids_to_add
                    )
                    processed_files += 1
                    logger.info(f"Arquivo '{file_name}' processado com sucesso ({len(docs_to_add)} chunks).")
                except Exception as e:
                    logger.error(f"Erro ao adicionar chunks de '{file_name}' ao ChromaDB: {e}")
                    continue

            except Exception as e:
                logger.error(f"Erro ao processar arquivo '{file_name}': {e}")
                continue

        logger.info(f"Ingestão concluída para '{collection_name}': {processed_files}/{len(files_to_process)} arquivos processados.")
        return processed_files

    async def _is_file_ingested(self, collection: chromadb.Collection, file_name: str) -> bool:
        """Verifica de forma assíncrona se um arquivo já foi ingerido."""
        try:
            result = await asyncio.to_thread(collection.get, where={"file": file_name}, limit=1)
            return bool(result['ids'])
        except Exception as e:
            logger.error(f"Erro ao verificar se arquivo '{file_name}' já foi ingerido: {e}")
            return False

    async def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[RetrievalResult]:
        """Recupera chunks semelhantes à query de uma coleção no ChromaDB."""
        try:
            collection = await asyncio.to_thread(self.db_client.get_collection, name=collection_name)
        except ValueError:
            raise ValueError(f"Coleção '{collection_name}' não encontrada.")
        
        embedding_results = await self.ai_processor.process_embedding_batch([query])
        query_embedding = embedding_results['results'][0].get('content')
        if not query_embedding:
            raise RuntimeError("Falha ao embeddar a query de busca.")

        results = await asyncio.to_thread(
            collection.query,
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieval_results = []
        if results['ids'] and results['ids'][0]:
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
                pass # Coleção não existia, o que está OK.
        else:
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.db_client = chromadb.PersistentClient(path=self.vector_store_path)
            logger.info("Todo o armazenamento vetorial foi limpo e recriado.")
