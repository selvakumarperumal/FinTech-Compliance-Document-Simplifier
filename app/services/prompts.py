"""
LLM Service for Compliance Document Simplification

This module provides multiple ChatPromptTemplate implementations using various LangChain core methods.
Choose the best approach for your specific use case.
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    StringPromptTemplate
)
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
)
from langchain_core.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ComplianceSimplifierPrompts:
    """Collection of different ChatPromptTemplate implementations for compliance document simplification"""
    
    # Base template content
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
    def basic_chat_prompt_template(cls) -> ChatPromptTemplate:
        """
        Method 1: Basic ChatPromptTemplate.from_template()
        Simple and straightforward approach
        """
        try:
            return ChatPromptTemplate.from_template(cls.SIMPLIFICATION_TEMPLATE)
        except Exception as e:
            logger.error(f"Failed to create basic_chat_prompt_template: {str(e)}")
            raise RuntimeError(f"Failed to create basic chat prompt template: {str(e)}")
    
    @classmethod
    def from_messages_template(cls) -> ChatPromptTemplate:
        """
        Method 2: ChatPromptTemplate.from_messages()
        Using separate system and human messages
        """
        try:
            system_message = SystemMessagePromptTemplate.from_template(
                "You are a compliance document simplifier. Your task is to continue simplifying regulatory or financial documents in plain, easy-to-understand English."
            )
        
            human_message = HumanMessagePromptTemplate.from_template(
                """
                The user has already simplified the previous part of the document.

                ---
                Previous Simplified Chunk:
                {previous_simplified}

                Current Original Chunk:
                {current_chunk}
                ---

                Task:
                Continue simplifying the current chunk in a tone and style consistent with the previous simplification. Do not repeat or restate content from earlier. Stay concise and clear. Use plain English suitable for someone with no legal or compliance background.

                Begin your continuation:
                """
                )
            
            return ChatPromptTemplate.from_messages([
                system_message,
                human_message
            ])
        except Exception as e:
            logger.error(f"Failed to create from_messages_template: {str(e)}")
            raise RuntimeError(f"Failed to create from messages template: {str(e)}")
    
    @classmethod
    def messages_placeholder_template(cls) -> ChatPromptTemplate:
        """
        Method 3: Using MessagesPlaceholder for conversation history
        Useful for maintaining conversation context
        """
        try:
            return ChatPromptTemplate.from_messages(
                [
                SystemMessagePromptTemplate.from_template(
                    "You are a compliance document simplifier. Continue simplifying documents in plain English, maintaining consistency with previous simplifications."
                ),
                MessagesPlaceholder(variable_name="conversation_history", optional=True),
                HumanMessagePromptTemplate.from_template(
                    """
                    Previous Simplified Chunk:
                    {previous_simplified}

                Current Original Chunk:
                {current_chunk}

                Continue simplifying the current chunk consistently. Use plain English suitable for non-experts.

                Begin your continuation:
                """
                )
            ]
        )
        except Exception as e:
            logger.error(f"Failed to create messages_placeholder_template: {str(e)}")
            raise RuntimeError(f"Failed to create messages placeholder template: {str(e)}")
    
    @classmethod
    def few_shot_template(cls, examples: Optional[List[Dict[str, str]]] = None) -> FewShotChatMessagePromptTemplate:
        """
        Method 5: FewShotChatMessagePromptTemplate with examples
        Provides examples to guide the model's behavior
        """
        if examples is None:
            examples = [
                {
                    "previous_simplified": "Banks must verify customer identity before opening accounts.",
                    "current_chunk": "Financial institutions shall implement robust customer due diligence procedures in accordance with regulatory requirements.",
                    "simplified_output": "Banks must thoroughly check who their customers are when opening new accounts."
                },
                {
                    "previous_simplified": "Companies must report suspicious activities to authorities.",
                    "current_chunk": "Entities are obligated to file Suspicious Activity Reports (SARs) with the Financial Crimes Enforcement Network within prescribed timeframes.",
                    "simplified_output": "Companies must file reports about suspicious activities with financial crime authorities within the required time limits."
                }
            ]
        
        example_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                "Previous: {previous_simplified}\nCurrent: {current_chunk}"
            ),
            AIMessage(content="{simplified_output}")
        ])
        
        return FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
            input_variables=["previous_simplified", "current_chunk"]
        )
    
    @classmethod
    def semantic_similarity_template(cls, vectorstore_cls: Optional[type] = None, 
                                   embeddings: Optional[Embeddings] = None) -> FewShotChatMessagePromptTemplate:
        """
        Method 6: Semantic similarity example selection
        Dynamically selects relevant examples based on input similarity
        Note: Requires vectorstore class (not instance) and embeddings
        """
        examples = [
            {
                "previous_simplified": "Banks must verify customer identity.",
                "current_chunk": "Enhanced due diligence procedures for high-risk customers.",
                "simplified_output": "Banks must do extra checks on risky customers."
            },
            {
                "previous_simplified": "Report suspicious transactions immediately.",
                "current_chunk": "Threshold reporting requirements for cash transactions.",
                "simplified_output": "Report large cash transactions over certain amounts."
            },
            {
                "previous_simplified": "Maintain accurate records.",
                "current_chunk": "Documentation retention policies and procedures.",
                "simplified_output": "Keep proper records and store them correctly."
            }
        ]
        
        if vectorstore_cls and embeddings:
            try:
                example_selector = SemanticSimilarityExampleSelector.from_examples(
                    examples,
                    embeddings,
                    vectorstore_cls,
                    k=2
                )
            except Exception:
                # Fallback to all examples if vectorstore setup fails
                example_selector = examples
        else:
            # Fallback to all examples if no vectorstore provided
            example_selector = examples
        
        example_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                "Previous: {previous_simplified}\nCurrent: {current_chunk}"
            ),
            AIMessage(content="{simplified_output}")
        ])
        
        return FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=example_selector if isinstance(example_selector, list) else None,
            example_selector=example_selector if not isinstance(example_selector, list) else None,
            input_variables=["previous_simplified", "current_chunk"]
        )
    
    @classmethod
    def custom_string_template(cls) -> StringPromptTemplate:
        """
        Method 7: Custom StringPromptTemplate
        For advanced customization and validation
        """
        try:
            class ComplianceStringTemplate(StringPromptTemplate):
                def format(self, **kwargs: Any) -> str:
                    try:
                        previous = kwargs.get("previous_simplified", "").strip()
                        current = kwargs.get("current_chunk", "").strip()
                        
                        # Custom validation
                        if not current:
                            raise ValueError("Current chunk cannot be empty")
                        
                        # Custom formatting logic
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
                            template = cls.SIMPLIFICATION_TEMPLATE
                        
                        return template.format(
                            previous_simplified=previous,
                            current_chunk=current
                        )
                    except Exception as e:
                        logger.error(f"Failed to format custom string template: {str(e)}")
                        raise RuntimeError(f"Failed to format template: {str(e)}")
                
                def format_messages(self, **kwargs: Any) -> List[Any]:
                    """
                    Add format_messages compatibility for testing
                    Returns a list with a single formatted string message
                    """
                    try:
                        formatted_text = self.format(**kwargs)
                        return [{"role": "user", "content": formatted_text}]
                    except Exception as e:
                        logger.error(f"Failed to format messages in custom template: {str(e)}")
                        raise RuntimeError(f"Failed to format messages: {str(e)}")
            
            return ComplianceStringTemplate(input_variables=["previous_simplified", "current_chunk"])
        except Exception as e:
            logger.error(f"Failed to create custom_string_template: {str(e)}")
            raise RuntimeError(f"Failed to create custom string template: {str(e)}")
    
    @classmethod
    def conditional_template(cls) -> ChatPromptTemplate:
        """
        Method 8: Conditional template based on document type
        Adapts based on input characteristics
        """
        return ChatPromptTemplate.from_template(
            """
            You are a compliance document simplifier.

            {%- if document_type == "regulatory" %}
            You are working with regulatory compliance documents. Focus on legal requirements and obligations.
            {%- elif document_type == "financial" %}
            You are working with financial compliance documents. Focus on financial procedures and requirements.
            {%- else %}
            You are working with general compliance documents.
            {%- endif %}

            ---
            Previous Simplified Chunk:
            {previous_simplified}

            Current Original Chunk:
            {current_chunk}

            {%- if chunk_complexity == "high" %}
            This is a complex section - take extra care to simplify technical terms.
            {%- endif %}
            ---

            Task:
            Continue simplifying the current chunk in a tone and style consistent with the previous simplification. Do not repeat or restate content from earlier. Stay concise and clear. Use plain English suitable for someone with no legal or compliance background.

            Begin your continuation:
            """,
            template_format="jinja2"
        )
    
    @classmethod
    def structured_output_template(cls) -> ChatPromptTemplate:
        """
        Method 9: Template for structured output
        Guides the model to produce structured responses
        """
        return ChatPromptTemplate.from_template(
            """
            You are a compliance document simplifier.

            ---
            Previous Simplified Chunk:
            {previous_simplified}

            Current Original Chunk:
            {current_chunk}
            ---

            Task:
            Simplify the current chunk and provide your response in the following structure:

            **Simplified Version:**
            [Your simplified text here]

            **Key Points:**
            - [Main point 1]
            - [Main point 2]
            - [Main point 3]

            **Plain English Summary:**
            [One sentence summary]

            Begin your structured response:
            """
            )
    
    @classmethod
    def multi_turn_conversation_template(cls) -> ChatPromptTemplate:
        """
        Method 10: Multi-turn conversation template
        For interactive simplification sessions
        """
        return ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a compliance document simplifier. You help users understand complex regulatory and financial documents by converting them into plain English.
                          You maintain conversation context and can:
                          1. Simplify document chunks sequentially
                          2. Answer questions about simplified content
                          3. Clarify confusing terms
                          4. Maintain consistency across the entire document"""),
            
            MessagesPlaceholder(variable_name="conversation_history", optional=True),
            
            HumanMessagePromptTemplate.from_template(
                """
                {%- if action == "simplify" %}
                Previous Simplified: {previous_simplified}
                Current Chunk: {current_chunk}

                Continue simplifying this chunk in plain English.
                {%- elif action == "clarify" %}
                Please clarify this term from the simplified text: {term_to_clarify}
                {%- elif action == "summarize" %}
                Provide a summary of all simplified content so far.
                {%- else %}
                {user_input}
                {%- endif %}
                """,
                template_format="jinja2")
                ])

# Example usage and testing functions
def demonstrate_all_templates():
    """Demonstrate all available prompt template methods"""
    prompts = ComplianceSimplifierPrompts()
    
    sample_data = {
        "previous_simplified": "Banks must verify customer identity before opening accounts.",
        "current_chunk": "Financial institutions shall implement enhanced due diligence procedures for politically exposed persons and high-risk customers in accordance with regulatory requirements."
    }
    
    methods = [
        ("Basic Template", prompts.basic_chat_prompt_template()),
        ("From Messages", prompts.from_messages_template()),
        ("Messages Placeholder", prompts.messages_placeholder_template()),
        ("Few Shot", prompts.few_shot_template()),
        ("Custom String", prompts.custom_string_template()),
        ("Conditional", prompts.conditional_template()),
        ("Structured Output", prompts.structured_output_template()),
        ("Multi-turn", prompts.multi_turn_conversation_template())
    ]
    
    print("=== Available Prompt Template Methods ===\n")
    
    for name, template in methods:
        print(f"🔹 {name}")
        try:
            if name == "Multi-turn":
                formatted = template.format_messages(
                    action="simplify",
                    **sample_data
                )
            elif name == "Custom String":
                # Custom String template uses format() method
                formatted = template.format(**sample_data)
            elif name == "Conditional":
                # Conditional template needs additional variables
                conditional_data = {
                    **sample_data,
                    "document_type": "regulatory",
                    "chunk_complexity": "high"
                }
                formatted = template.format_messages(**conditional_data)
            else:
                formatted = template.format_messages(**sample_data)
            
            print(f"✅ Template successfully created and formatted")
            print(f"📝 Input variables: {getattr(template, 'input_variables', 'N/A')}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)


if __name__ == "__main__":
    demonstrate_all_templates()
