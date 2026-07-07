from langchain_nvidia_ai_endpoints import ChatNVIDIA
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
        """Initialize the LLM service with the specified NVIDIA model."""
        self.model = model
        self.client = ChatNVIDIA(
            model=self.model,
            api_key=settings.nvidia_api_key.get_secret_value(),
            temperature=0.1,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500,
        )
        self.prompt = ComplianceSimplifierPrompts.custom_string_template()
        logger.info(f"LLMService initialized with NVIDIA model: {self.model}")

    async def lcel_for_simplification(self, previous_simplified: str, current_chunk: str) -> str:
        """Perform LCEL chain for content simplification."""
        lcels = RunnableSequence(self.prompt, self.client)
        response = await lcels.ainvoke({
            "previous_simplified": previous_simplified,
            "current_chunk": current_chunk
        })
        return response.content

    def _generate_fallback_summary(self, text: str) -> str:
        """Generate a simulated simplified summary when LLM fails (e.g., offline/no internet)."""
        lines = text.split('\n')
        summary = []
        summary.append("### [Offline Preview Mode] Simplified Compliance Summary")
        summary.append("> Note: The NVIDIA LLM service is currently offline or unreachable. Displaying fallback rule-based summary preview.")
        summary.append("")
        summary.append("#### Key Compliance Requirements Identified:")
        
        count = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            lower_line = line.lower()
            if any(k in lower_line for k in ["shall", "must", "require", "comply", "obligation", "standard", "rule", "fee", "action", "penalty"]):
                if 20 < len(line) < 200:
                    summary.append(f"- **Requirement**: {line}")
                    count += 1
            if count >= 6:
                break
                
        if count == 0:
            summary.append("- **General Requirement**: Entities must adhere to all stated guidelines, maintain appropriate documentation, and ensure internal controls are active.")
            summary.append("- **Audit Log**: Keep detailed logs of all compliance actions and revisions for audit checks.")
            summary.append("- **Security Standard**: Ensure transport encryption and access controls are fully enforced.")
            
        summary.append("\n#### Plain Language Translation:")
        summary.append("1. **Adhere to Obligations**: Carefully follow the detailed rules above.")
        summary.append("2. **Document Action**: Ensure all transactions and adjustments are log-verifiable.")
        summary.append("3. **Regular Auditing**: Periodically inspect records for accuracy and security compliance.")
        
        return "\n".join(summary)

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
                logger.error(f"Failed to process chunk via LLM: {repr(e)}. Using fallback preview summarizer.")
                fallback_response = self._generate_fallback_summary(chunk_text)
                responses.append(fallback_response)
                previous_simplified = fallback_response

        return "\n".join(responses)


async def simplify_content_service(content: List[Document], model: str = None) -> dict:
    """Service function to simplify content."""
    if model is None or model.strip() == "" or model.strip().lower() == "string":
        model = settings.nvidia_model_name
    
    llm_service = LLMService(model=model)
    simplified = await llm_service.simplify_content(content)
    return {"simplified_content": simplified}

