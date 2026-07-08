from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.prompts import ComplianceSimplifierPrompts
from app.config import settings
from langchain_core.runnables import RunnableSequence
from typing import List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self, model: str):
        """Initialize the LLM service with the specified Google Gemini model."""
        self.model = model
        api_key = settings.google_api_key.get_secret_value() if settings.google_api_key else None
        self.client = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=api_key,
            temperature=0.1,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500,
        )
        self.prompt = ComplianceSimplifierPrompts.custom_string_template()
        logger.info(f"LLMService initialized with Google Gemini model: {self.model}")

    async def lcel_for_simplification(self, previous_simplified: str, current_chunk: str) -> str:
        """Perform LCEL chain for content simplification."""
        lcels = RunnableSequence(self.prompt, self.client)
        response = await lcels.ainvoke({
            "previous_simplified": previous_simplified,
            "current_chunk": current_chunk
        })
        return response.content

    async def simplify_content(self, content: List[Document]) -> str:
        """Simplify the provided content using the LLM."""
        chunks = self.text_splitter.split_documents(content)
        logger.info(f"Content split into {len(chunks)} chunks for simplification.")
        
        responses = []
        previous_simplified = ""
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Simplifying chunk {i+1}/{len(chunks)}")
            
            chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            try:
                response = await self.lcel_for_simplification(previous_simplified, chunk_text)
                responses.append(response)
                previous_simplified = response
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1} via LLM: {str(e)}")
                raise RuntimeError(f"Failed to process chunk {i+1} via LLM: {str(e)}")

        return "\n".join(responses)


async def simplify_content_service(content: List[Document], model: str = None) -> dict:
    """Service function to simplify content."""
    if model is None or model.strip() == "" or model.strip().lower() == "string":
        model = settings.google_model_name
    
    llm_service = LLMService(model=model)
    simplified = await llm_service.simplify_content(content)
    return {"simplified_content": simplified}

