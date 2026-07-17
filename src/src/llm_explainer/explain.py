import os
import logging
from typing import List, Optional
from pydantic import BaseModel

# If using OpenAI or MCP-compatible models
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    emotions: List[str]
    rag_context: str
    explanation: str


class EmotionLLMExplainer:
    """
    Generates an interpretable explanation for predicted emotions by
    merging model predictions with psychology-backed context (RAG).
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Parameters
        ----------
        model_name : str
            Any MCP/LLM model name (GPT-4o, GPT-4.1, Claude, Llama-3 via API).
        """
        self.model_name = model_name

    def explain_emotions(self, emotions: List[str], rag_context: str) -> str:
        """
        Uses an LLM to produce a natural-language explanation of the detected emotions.

        Parameters
        ----------
        emotions : list of str
            CNN model predictions (e.g., ['sad', 'fear'])
        rag_context : str
            Text retrieved from psychological studies via RAG

        Returns
        -------
        str : Human-readable explanation
        """

        system_prompt = """
        You are an Emotion AI Explainer. Your job is to interpret human
        emotions in a psychologically accurate and empathetic way.
        
        Use:
        - The predicted emotions (from CNN)
        - The RAG psychology context (behavioral science)
        
        Produce a clear, friendly explanation of what these emotions may mean
        and how someone experiencing them might be feeling internally.
        """

        user_prompt = f"""
        Detected Emotions: {emotions}
        Psychology Context from RAG: {rag_context}

        Combine these into a detailed but readable explanation
        suitable for a mobile mental-health app or wellness dashboard.
        """

        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=350,
                temperature=0.6
            )

            explanation = response.choices[0].message.content
            return explanation

        except Exception:
            logger.exception("LLM explanation generation failed")
            return "Unable to generate explanation at this time."

