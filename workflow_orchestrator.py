import asyncio
import logging
from typing import List, Dict, Any, Optional

# Importa as classes dos módulos de núcleo
from .ai_processor import AIProcessor
from .rag_manager import RAGManager, RetrievalResult

logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    """
    Orquestrador de alto nível para fluxos de trabalho (workflows) de IA.

    Esta classe combina as funcionalidades dos módulos de núcleo (AIProcessor, RAGManager)
    para executar lógicas de negócio complexas, como classificação híbrida ou
    enriquecimento em lote via RAG. Isso mantém os notebooks de execução limpos
    e a lógica de negócio centralizada, testável e reutilizável.
    """

    def __init__(self, ai_processor: AIProcessor, rag_manager: RAGManager):
        """
        Inicializa o orquestrador com as instâncias dos módulos necessários.

        Args:
            ai_processor (AIProcessor): Instância do processador de IA para chamadas de API.
            rag_manager (RAGManager): Instância do gerenciador de RAG para busca de similaridade.
        """
        if not isinstance(ai_processor, AIProcessor):
            raise TypeError("ai_processor deve ser uma instância de AIProcessor.")
        if not isinstance(rag_manager, RAGManager):
            raise TypeError("rag_manager deve ser uma instância de RAGManager.")
            
        self.processor = ai_processor
        self.rag = rag_manager
        logger.info("WorkflowOrchestrator inicializado e pronto para executar workflows.")

    async def process_batch_with_rag(
        self,
        texts: List[str],
        prompt_template: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        batch_id: Optional[str] = "rag_batch_process"
    ) -> Dict[str, Any]:
        """
        Workflow: Processa um lote de textos, enriquecendo cada um com contexto via RAG.

        Para cada texto na lista, este método realiza uma busca de similaridade,
        recupera os contextos mais relevantes e os injeta no prompt antes de
        enviar para processamento em lote pela IA.

        Args:
            texts (List[str]): A lista de textos ou queries a serem processados.
            prompt_template (str): O template do prompt. Deve conter `{context}` e `{text}`.
            top_k (int): O número máximo de chunks de contexto a serem recuperados para cada texto.
            similarity_threshold (float): O score mínimo de similaridade para um chunk ser considerado.
            batch_id (Optional[str]): Um ID customizado para o lote de processamento da IA.

        Returns:
            Dict[str, Any]: O resultado do `AIProcessor.process_batch`, incluindo os resultados
                            processados e as estatísticas do lote.
        """
        logger.info(f"Iniciando workflow 'process_batch_with_rag' para {len(texts)} itens.")
        
        # Cria uma tarefa de enriquecimento para cada texto no lote.
        enrichment_tasks = []
        for text in texts:
            task = self._enrich_single_prompt(text, prompt_template, top_k, similarity_threshold)
            enrichment_tasks.append(task)
        
        # Executa o enriquecimento de todos os prompts em paralelo.
        enriched_prompts = await asyncio.gather(*enrichment_tasks)

        # Filtra prompts que não puderam ser enriquecidos (se houver).
        final_prompts = [p for p in enriched_prompts if p is not None]

        if not final_prompts:
            logger.warning("Nenhum prompt pôde ser enriquecido. O processamento da IA será pulado.")
            return {"results": [], "batch_stats": None, "batch_id": batch_id}

        # Processa o lote de prompts enriquecidos com o AIProcessor.
        # O template agora é simples, pois o texto já foi formatado.
        results = await self.processor.process_batch(
            texts=final_prompts,
            prompt_template="{text}", 
            batch_id=batch_id
        )

        logger.info(f"Workflow 'process_batch_with_rag' concluído.")
        return results

    async def _enrich_single_prompt(self, text: str, prompt_template: str, top_k: int, similarity_threshold: float) -> Optional[str]:
        """Função auxiliar para recuperar e formatar um único prompt."""
        try:
            retrieved_results = await self.rag.retrieve(text, top_k=top_k, similarity_threshold=similarity_threshold)
            
            # Constrói o contexto a partir dos resultados recuperados.
            context_str = "\n".join([res.content for res in retrieved_results])
            
            # Formata o prompt final.
            return prompt_template.format(context=context_str, text=text)
        except Exception as e:
            logger.error(f"Falha ao enriquecer prompt para o texto '{text[:50]}...': {e}")
            return None

    async def classify_with_similarity_fallback(
        self,
        texts: List[str],
        prompt_template_for_ia: str,
        similarity_threshold: float = 0.8,
        top_k: int = 1,
        batch_id: Optional[str] = "hybrid_classification_batch"
    ) -> List[Dict[str, Any]]:
        """
        Workflow: Classifica um lote de textos usando um método híbrido.

        Primeiro, tenta classificar cada texto com base na similaridade com documentos
        de referência. Se a similaridade for baixa (abaixo do threshold), o texto é
        enviado para a IA para uma classificação mais robusta.

        Args:
            texts (List[str]): A lista de textos a serem classificados.
            prompt_template_for_ia (str): O prompt a ser usado para a IA (ex: "Classifique: {text}").
            similarity_threshold (float): O score de similaridade mínimo para uma classificação confiável.
            top_k (int): O número de resultados de similaridade a considerar (geralmente 1 para classificação).
            batch_id (Optional[str]): Um ID customizado para o lote enviado à IA.

        Returns:
            List[Dict[str, Any]]: Uma lista de dicionários, cada um representando o resultado da
                                   classificação para um texto.
        """
        logger.info(f"Iniciando workflow 'classify_with_similarity_fallback' para {len(texts)} itens.")
        
        final_results = []
        texts_for_ia = []
        original_indices_for_ia = {}

        # 1. Tenta classificar via similaridade primeiro.
        for i, text in enumerate(texts):
            try:
                retrieved = await self.rag.retrieve(text, top_k=top_k, similarity_threshold=0.0) # Pega o melhor resultado, sem filtro
                
                max_similarity = retrieved[0].similarity if retrieved else 0.0
                
                if max_similarity >= similarity_threshold:
                    # Classificação por similaridade foi bem-sucedida.
                    best_match = retrieved[0]
                    category = best_match.metadata.get('file', 'unknown').split('.')[0]
                    final_results.append({
                        'text': text,
                        'classification': category,
                        'method': 'similarity',
                        'confidence': max_similarity
                    })
                else:
                    # Baixa confiança, marcar para envio à IA.
                    texts_for_ia.append(text)
                    original_indices_for_ia[text] = i # Guarda o índice original
                    final_results.append(None) # Marcador de posição
            except Exception as e:
                logger.error(f"Erro ao buscar similaridade para o texto '{text[:50]}...': {e}. Marcando para IA.")
                texts_for_ia.append(text)
                original_indices_for_ia[text] = i
                final_results.append(None)

        # 2. Processa em lote apenas os textos que precisam da IA.
        if texts_for_ia:
            logger.info(f"{len(texts_for_ia)}/{len(texts)} textos marcados para classificação via IA.")
            ia_results = await self.processor.process_batch(
                texts=texts_for_ia,
                prompt_template=prompt_template_for_ia,
                batch_id=batch_id
            )

            # 3. Integra os resultados da IA de volta na lista final.
            for i, result in enumerate(ia_results['results']):
                original_text = texts_for_ia[i]
                original_idx = original_indices_for_ia[original_text]
                
                if result['success']:
                    final_results[original_idx] = {
                        'text': original_text,
                        'classification': result['content'],
                        'method': 'ia',
                        'confidence': None # Confiança não aplicável para IA
                    }
                else:
                    final_results[original_idx] = {
                        'text': original_text,
                        'classification': 'error',
                        'method': 'ia_failed',
                        'confidence': None,
                        'error_details': result.get('error')
                    }
        
        logger.info("Workflow 'classify_with_similarity_fallback' concluído.")
        return final_results
