from langchain_anthropic.chat_models import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.prompts import ComplianceSimplifierPrompts
from app.config import settings
from langchain_core.runnables import RunnableSequence
from typing import Optional, List
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model: str):
        """Initialize the LLM service with the specified model."""
        try:
            self.model = model
            self.client = ChatAnthropic(
                model_name=self.model,
                api_key=settings.anthropic_api_key.get_secret_value(),
                temperature=0.1,
                timeout=None,
                max_retries=3,
                stop=None,
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000,
                chunk_overlap=500,
            )
            self.prompt = ComplianceSimplifierPrompts.custom_string_template()

            logger.info(f"LLMService initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLMService with model {model}: {str(e)}")
            raise RuntimeError(f"Failed to initialize LLM service: {str(e)}")

    async def lcel_for_simplification(self, previous_simplified:str, current_chunk:str) -> str:
        """Perform Local Contextualized Explanation Learning (LCEL) for content simplification."""
        try:
            # Create a runnable sequence for LCEL
            lcels = RunnableSequence(
                self.prompt,
                self.client,
            )
            
            # Run the LCEL process
            response = await lcels.ainvoke({
                "previous_simplified": previous_simplified,
                "current_chunk": current_chunk
            })

            return response.content
        except Exception as e:
            logger.error(f"Failed in lcel_for_simplification: {str(e)}")
            raise RuntimeError(f"Failed to simplify content chunk: {str(e)}")

    async def simplify_content(self, content: List[Document]) -> dict[str, str]:
        """Simplify the provided content using the LLM."""
        try:
            try:
                chunks = self.text_splitter.split_documents(content)
            except Exception as e:
                logger.error(f"Error splitting content into chunks: {str(e)}")
                raise ValueError(f"Error splitting content: {e}")

            logger.info(f"Content split into {len(chunks)} chunks for simplification.")
            responses = []
            
            previous_simplified = ""
            # Process each chunk with LCEL
            for chunk in chunks:
                try:
                    # Convert Document to string for processing
                    chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
                    response = await self.lcel_for_simplification(previous_simplified, chunk_text)
                    responses.append(response)
                    previous_simplified = response
                except Exception as e:
                    logger.error(f"Failed to process chunk: {str(e)}")
                    continue

            return "\n".join(responses)
        except ValueError as ve:
            logger.error(f"Validation error in simplify_content: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in simplify_content: {str(e)}")
            raise RuntimeError(f"Failed to simplify content: {str(e)}")

async def simplify_content_service(content: List[Document], model: Optional[str] = None) -> str:
    """Service function to simplify content."""
    try:
        if model is None:
            logger.warning("No model specified for simplify_content_service, using default")
            model = "claude-3-haiku-20240307"  # Default model
        
        llm_service = LLMService(model=model)
        return {"simplified_content": await llm_service.simplify_content(content)}
    except Exception as e:
        logger.error(f"Failed in simplify_content_service: {str(e)}")
        raise RuntimeError(f"Failed to simplify content service: {str(e)}")



