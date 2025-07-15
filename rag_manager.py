import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Constantes padrão
DEFAULT_CHUNK_SIZE = 512  # Tamanho máximo de chunk em caracteres (ajustável para evitar limites de token)
DEFAULT_SIMILARITY_THRESHOLD = 0.7  # Threshold mínimo para considerar um match
DEFAULT_TOP_K = 5  # Número padrão de resultados a retornar

@dataclass
class RetrievalResult:
    """Estrutura para resultados de recuperação."""
    content: str  # Texto do chunk
    similarity: float  # Score de similaridade cosseno
    metadata: Dict[str, Any]  # Ex: {'file': 'arquivo.txt', 'chunk_id': 0}

class RAGManager:
    """
    Módulo para Retrieval-Augmented Generation (RAG) local, integrado ao AIProcessor.
    Gerencia ingestão de documentos, armazenamento vetorial simples e recuperação para enriquecer prompts.
    """

    def __init__(self, ai_processor: 'AIProcessor', project_dir: str):
        """
        Inicializa o RAGManager.
        - ai_processor: Instância do AIProcessor para embeddings e (opcionalmente) completions.
        - project_dir: Caminho para a pasta do projeto (ex: 'notebooks/teste/').
        """
        self.ai_processor = ai_processor
        self.project_dir = os.path.abspath(project_dir)
        self.documents_dir = os.path.join(self.project_dir, 'documents')
        self.vector_store_dir = os.path.join(self.project_dir, 'vector_store')
        
        # Cria diretórios se não existirem
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vector_store_dir, exist_ok=True)
        
        self.vector_store: Dict[str, List[Dict[str, Any]]] = {}  # Armazenamento in-memory: {file: [{'chunk': text, 'embedding': list[float], 'metadata': dict}]}
        
        logger.info(f"RAGManager inicializado para projeto em '{self.project_dir}'.")

    async def ingest_documents(self, relative_path: str = '', force_update: bool = False, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
        """
        Ingestão de documentos: lê arquivos de 'documents/', chunkifica, embedda e armazena localmente.
        - relative_path: Subpasta dentro de 'documents/' (opcional).
        - force_update: Força re-embedding mesmo se já existir.
        - chunk_size: Tamanho máximo de cada chunk em caracteres.
        Retorna o número de arquivos processados.
        """
        ingest_dir = os.path.join(self.documents_dir, relative_path)
        if not os.path.exists(ingest_dir):
            raise FileNotFoundError(f"Diretório '{ingest_dir}' não encontrado.")
        
        processed_count = 0
        files = [f for f in os.listdir(ingest_dir) if os.path.isfile(os.path.join(ingest_dir, f))]
        
        for file_name in files:
            file_path = os.path.join(ingest_dir, file_name)
            store_path = os.path.join(self.vector_store_dir, f"{file_name}.json")
            
            if not force_update and os.path.exists(store_path):
                self._load_from_store(file_name, store_path)
                processed_count += 1
                continue
            
            # Lê o arquivo (assume texto; para outros formatos, pré-processe no notebook)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Chunking flexível
            chunks = self._chunk_text(content, chunk_size)
            
            # Embedda os chunks usando AIProcessor
            embeddings = await self.ai_processor.process_embedding_batch(chunks, batch_id=f"ingest_{file_name}", custom_ids=[f"{file_name}_chunk_{i}" for i in range(len(chunks))])
            
            # Armazena
            vector_data = []
            for i, emb_result in enumerate(embeddings['results']):
                if emb_result['success']:
                    vector_data.append({
                        'chunk': chunks[i],
                        'embedding': emb_result['content'],
                        'metadata': {'file': file_name, 'chunk_id': i}
                    })
            
            self.vector_store[file_name] = vector_data
            self._save_to_store(file_name, store_path, vector_data)
            processed_count += 1
        
        logger.info(f"Ingestão concluída: {processed_count} arquivos processados.")
        return processed_count

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Chunking simples: divide texto em pedaços de até 'chunk_size' caracteres, respeitando frases."""
        chunks = []
        current_chunk = ""
        for sentence in text.split('. '):  # Divide por frases para preservar contexto
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _save_to_store(self, file_name: str, store_path: str, vector_data: List[Dict[str, Any]]):
        """Salva embeddings em JSON."""
        with open(store_path, 'w', encoding='utf-8') as f:
            json.dump(vector_data, f, ensure_ascii=False, indent=2)
        logger.debug(f"Embeddings salvos para '{file_name}' em '{store_path}'.")

    def _load_from_store(self, file_name: str, store_path: str):
        """Carrega embeddings de JSON para memória."""
        with open(store_path, 'r', encoding='utf-8') as f:
            vector_data = json.load(f)
        self.vector_store[file_name] = vector_data
        logger.debug(f"Embeddings carregados para '{file_name}' de '{store_path}'.")

    async def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> List[RetrievalResult]:
        """
        Recupera chunks semelhantes à query.
        - query: Texto da consulta.
        - top_k: Número máximo de resultados.
        - similarity_threshold: Score mínimo para inclusão.
        Retorna lista ordenada por similaridade.
        """
        if not self.vector_store:
            raise ValueError("Nenhum documento ingerido. Execute ingest_documents primeiro.")
        
        # Embedda a query
        query_emb_result = await self.ai_processor.process_embedding_batch([query], batch_id="retrieve_query")
        if not query_emb_result['results'][0]['success']:
            raise RuntimeError("Falha ao embeddar a query.")
        query_embedding = query_emb_result['results'][0]['content']
        
        # Busca vetorial simples: calcula similaridade cosseno para todos os chunks
        results = []
        for file_name, vectors in self.vector_store.items():
            for vec in vectors:
                sim = self._cosine_similarity(query_embedding, vec['embedding'])
                if sim >= similarity_threshold:
                    results.append(RetrievalResult(content=vec['chunk'], similarity=sim, metadata=vec['metadata']))
        
        # Ordena e retorna top-k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Implementação manual de similaridade cosseno (sem libs externas)."""
        if len(vec1) != len(vec2):
            return 0.0
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a ** 2 for a in vec1) ** 0.5
        norm2 = sum(b ** 2 for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def enrich_prompts(self, base_prompts: List[str], retrieved: List[RetrievalResult], context_template: str = "Contexto relevante: {context}\n") -> List[str]:
        """
        Enriquece uma lista de prompts com contextos recuperados (para RAG).
        - base_prompts: Lista de prompts base.
        - retrieved: Resultados de retrieve().
        - context_template: Formato para inserir o contexto.
        Retorna prompts enriquecidos.
        """
        if len(base_prompts) != 1 and len(base_prompts) != len(retrieved):
            raise ValueError("Número de prompts deve ser 1 (para todos) ou igual ao de retrieved.")
        
        contexts = [context_template.format(context=res.content) for res in retrieved]
        combined_context = "\n".join(contexts)
        
        enriched = []
        for prompt in base_prompts:
            enriched.append(combined_context + prompt)
        
        return enriched

    def clear_vector_store(self, file_name: Optional[str] = None):
        """Limpa o armazenamento vetorial (memória e disco) para um arquivo ou todos."""
        if file_name:
            if file_name in self.vector_store:
                del self.vector_store[file_name]
            store_path = os.path.join(self.vector_store_dir, f"{file_name}.json")
            if os.path.exists(store_path):
                os.remove(store_path)
        else:
            self.vector_store.clear()
            for f in os.listdir(self.vector_store_dir):
                os.remove(os.path.join(self.vector_store_dir, f))
        logger.info("Armazenamento vetorial limpo.")
