"""
LLM Service for Compliance Document Simplification

This module provides ChatPromptTemplate implementation for compliance document simplification.
"""

from langchain_core.prompts import StringPromptTemplate
from typing import List, Any
import logging

logger = logging.getLogger(__name__)


class ComplianceSimplifierPrompts:
    """Prompt templates for compliance document simplification"""
    
    SIMPLIFICATION_TEMPLATE = """
    You are a compliance document simplifier.

    Your task is to continue simplifying a long regulatory or financial document in plain, easy-to-understand English. The user has already simplified the previous part of the document.

    ---
    Previous Simplified Chunk:
    {previous_simplified}

    Current Original Chunk:
    {current_chunk}
    ---

    Task:
    Continue simplifying the current chunk in a tone and style consistent with the previous simplification. Do not repeat or restate content from earlier. Stay concise and clear. Use plain English suitable for someone with no legal or compliance background.
    Only produce simplification for current chunk.

    Begin your continuation:
    """

    @classmethod
    def custom_string_template(cls) -> StringPromptTemplate:
        """Custom StringPromptTemplate for compliance document simplification."""
        try:
            class ComplianceStringTemplate(StringPromptTemplate):
                def format(self, **kwargs: Any) -> str:
                    previous = kwargs.get("previous_simplified", "")
                    if isinstance(previous, list):
                        parts = []
                        for part in previous:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(part["text"])
                        previous = "".join(parts)
                    elif not isinstance(previous, str):
                        previous = str(previous)
                    previous = previous.strip()

                    current = kwargs.get("current_chunk", "")
                    if isinstance(current, list):
                        parts = []
                        for part in current:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(part["text"])
                        current = "".join(parts)
                    elif not isinstance(current, str):
                        current = str(current)
                    current = current.strip()
                    
                    if not current:
                        raise ValueError("Current chunk cannot be empty")
                    
                    if not previous:
                        template = """
                        You are a compliance document simplifier.

                        Document Chunk to Simplify:
                        {current_chunk}

                        Task:
                        Simplify this regulatory or financial document chunk in plain, easy-to-understand English. Use simple words suitable for someone with no legal or compliance background.

                        Begin your simplification:
                        """
                    else:
                        template = ComplianceSimplifierPrompts.SIMPLIFICATION_TEMPLATE
                    
                    return template.format(
                        previous_simplified=previous,
                        current_chunk=current
                    )
                
                def format_messages(self, **kwargs: Any) -> List[Any]:
                    """Returns a list with a single formatted string message."""
                    formatted_text = self.format(**kwargs)
                    return [{"role": "user", "content": formatted_text}]
            
            return ComplianceStringTemplate(input_variables=["previous_simplified", "current_chunk"])
        except Exception as e:
            logger.error(f"Failed to create custom_string_template: {str(e)}")
            raise RuntimeError(f"Failed to create custom string template: {str(e)}")

