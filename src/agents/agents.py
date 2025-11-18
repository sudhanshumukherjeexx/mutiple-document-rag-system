"""
RAG agents module for self-corrected RAG pipeline.
Implements Guardrail, Generation, and Evaluation agents.
"""

from typing import List, Tuple
import asyncio
import logging

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..config_loader import config
from ..metrics import Timer

logger = logging.getLogger(__name__)


# Pydantic Models for Structured Outputs

class GuardrailCheck(BaseModel):
    """Schema for the Guardrail Agent's relevance check."""
    is_relevant: bool = Field(
        description="True if the context is relevant to the question, False otherwise."
    )
    justification: str = Field(
        description="A brief explanation for the relevance decision."
    )


class Evaluation(BaseModel):
    """Schema for the Evaluator Agent's factual consistency score."""
    score: int = Field(
        description="A score from 1 (poor) to 5 (perfect) for factual consistency.",
        ge=1,
        le=5
    )
    justification: str = Field(
        description="A brief explanation for the score."
    )


# Agent Classes

class GuardrailAgent:
    """Agent that filters irrelevant context for RAG."""
    
    def __init__(self):
        """Initialize the Guardrail Agent."""
        model_name = config.get('models.guardrail.name')
        temperature = config.get('models.guardrail.temperature')
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=config.get('models.guardrail.max_tokens'),
            timeout=config.get('api.request_timeout')
        )
        
        self.llm_structured = self.llm.with_structured_output(GuardrailCheck)
        
        self.prompt = PromptTemplate(
            template="""You are a 'Guardrail Agent' acting as a relevance filter.
Your job is to determine if the following CONTEXT is relevant for answering the user's QUESTION.

Guidelines:
- Respond with 'is_relevant: true' if the context contains information that can help answer the question
- Respond with 'is_relevant: false' if the context is completely unrelated
- Be strict but reasonable - partial relevance should be marked as relevant
- Consider semantic similarity, not just keyword matching

QUESTION:
{question}

CONTEXT:
{context}

Provide your assessment:""",
            input_variables=["question", "context"]
        )
        
        self.chain = self.prompt | self.llm_structured
        
        logger.info(f"GuardrailAgent initialized with model: {model_name}")
    
    async def check_relevance(self, question: str, context: str) -> GuardrailCheck:
        """
        Check if context is relevant to the question.
        
        Args:
            question: User's question
            context: Context to evaluate
            
        Returns:
            GuardrailCheck result
        """
        with Timer("Guardrail check"):
            result = await self.chain.ainvoke({
                "question": question,
                "context": context
            })
        
        return result
    
    async def filter_documents(
        self,
        question: str,
        documents: List[Document],
        parallel: bool = True
    ) -> Tuple[List[Document], List[str]]:
        """
        Filter documents for relevance.
        
        Args:
            question: User's question
            documents: List of documents to filter
            parallel: Whether to process documents in parallel
            
        Returns:
            Tuple of (relevant_documents, justifications)
        """
        if not documents:
            return [], []
        
        if parallel and len(documents) > 1:
            return await self._filter_parallel(question, documents)
        else:
            return await self._filter_sequential(question, documents)
    
    async def _filter_sequential(
        self,
        question: str,
        documents: List[Document]
    ) -> Tuple[List[Document], List[str]]:
        """Filter documents sequentially."""
        relevant_docs = []
        justifications = []
        
        for i, doc in enumerate(documents):
            logger.debug(f"Checking document {i+1}/{len(documents)}")
            
            try:
                result = await self.check_relevance(question, doc.page_content)
                
                if result.is_relevant:
                    relevant_docs.append(doc)
                    justifications.append(result.justification)
                    logger.debug(f"Document {i+1} is RELEVANT: {result.justification}")
                else:
                    logger.debug(f"Document {i+1} is IRRELEVANT: {result.justification}")
                    
            except Exception as e:
                logger.warning(f"Error checking document {i+1}: {e}. Including it by default.")
                relevant_docs.append(doc)
                justifications.append(f"Error during check: {e}")
        
        return relevant_docs, justifications
    
    async def _filter_parallel(
        self,
        question: str,
        documents: List[Document]
    ) -> Tuple[List[Document], List[str]]:
        """Filter documents in parallel for better performance."""
        logger.debug(f"Checking {len(documents)} documents in parallel")
        
        async def check_doc(doc: Document, index: int):
            try:
                result = await self.check_relevance(question, doc.page_content)
                return (index, doc, result)
            except Exception as e:
                logger.warning(f"Error checking document {index+1}: {e}")
                return (index, doc, GuardrailCheck(
                    is_relevant=True,
                    justification=f"Error during check: {e}"
                ))
        
        # Check all documents concurrently
        tasks = [check_doc(doc, i) for i, doc in enumerate(documents)]
        results = await asyncio.gather(*tasks)
        
        # Filter relevant documents
        relevant_docs = []
        justifications = []
        
        for index, doc, result in sorted(results, key=lambda x: x[0]):
            if result.is_relevant:
                relevant_docs.append(doc)
                justifications.append(result.justification)
                logger.debug(f"Document {index+1} is RELEVANT: {result.justification}")
            else:
                logger.debug(f"Document {index+1} is IRRELEVANT: {result.justification}")
        
        return relevant_docs, justifications


class GenerationAgent:
    """Agent that generates answers from context."""
    
    def __init__(self):
        """Initialize the Generation Agent."""
        model_name = config.get('models.generate.name')
        temperature = config.get('models.generate.temperature')
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=config.get('models.generate.max_tokens'),
            timeout=config.get('api.request_timeout')
        )
        
        self.prompt = PromptTemplate(
            template="""You are a helpful AI assistant. Answer the user's QUESTION based *only* on the following SOURCE CONTEXT.

Important guidelines:
- Use ONLY the information provided in the SOURCE CONTEXT
- Do not use any outside information or knowledge
- If the context is not sufficient to answer the question, clearly state that
- Be accurate, concise, and well-structured
- Cite specific parts of the context when relevant

QUESTION:
{question}

SOURCE CONTEXT:
{context}

ANSWER:""",
            input_variables=["question", "context"]
        )
        
        self.chain = self.prompt | self.llm
        
        logger.info(f"GenerationAgent initialized with model: {model_name}")
    
    async def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer from the given context.
        
        Args:
            question: User's question
            context: Source context for answering
            
        Returns:
            Generated answer
        """
        with Timer("Answer generation"):
            result = await self.chain.ainvoke({
                "question": question,
                "context": context
            })
        
        answer = result.content
        logger.debug(f"Generated answer: {answer[:100]}...")
        
        return answer


class EvaluationAgent:
    """Agent that evaluates answer quality."""
    
    def __init__(self):
        """Initialize the Evaluation Agent."""
        model_name = config.get('models.evaluate.name')
        temperature = config.get('models.evaluate.temperature')
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=config.get('models.evaluate.max_tokens'),
            timeout=config.get('api.request_timeout')
        )
        
        self.llm_structured = self.llm.with_structured_output(Evaluation)
        
        self.prompt = PromptTemplate(
            template="""You are an 'Evaluator Agent' assessing the factual consistency of a generated ANSWER.
Your job is to determine if the ANSWER is fully supported by the given SOURCE CONTEXT.

Scoring guidelines:
- Score 5 (Perfect): The ANSWER is fully and verifiably supported by the SOURCE CONTEXT with no hallucinations
- Score 4 (Excellent): The ANSWER is mostly supported with only very minor unsupported details
- Score 3 (Good): The ANSWER is partially supported but contains some unsupported information
- Score 2 (Poor): The ANSWER contains significant information not present in the SOURCE CONTEXT
- Score 1 (Very Poor): The ANSWER is mostly or entirely unsupported by the SOURCE CONTEXT (hallucination)

Additional considerations:
- Answers that honestly state "information not available" should receive high scores if accurate
- Check for factual accuracy, not just semantic similarity
- Be strict but fair

SOURCE CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Provide your evaluation:""",
            input_variables=["context", "answer"]
        )
        
        self.chain = self.prompt | self.llm_structured
        
        logger.info(f"EvaluationAgent initialized with model: {model_name}")
    
    async def evaluate_answer(self, answer: str, context: str) -> Evaluation:
        """
        Evaluate the factual consistency of an answer.
        
        Args:
            answer: Generated answer to evaluate
            context: Source context used for generation
            
        Returns:
            Evaluation result
        """
        with Timer("Answer evaluation"):
            result = await self.chain.ainvoke({
                "context": context,
                "answer": answer
            })
        
        logger.debug(f"Evaluation: score={result.score}/5, justification={result.justification}")
        
        return result
