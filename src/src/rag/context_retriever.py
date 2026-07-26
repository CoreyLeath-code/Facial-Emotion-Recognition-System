class EmotionContextRetriever:
    """
    Local psychology-context retriever with an opt-in Pinecone backend.

    The local dictionary remains the deterministic default. Pinecone is used only
    when EMOTION_CONTEXT_BACKEND=pinecone and the hosted index is configured.
    """

    def __init__(self) -> None:
        self.backend = os.getenv("EMOTION_CONTEXT_BACKEND", "local").strip().lower()
        if self.backend not in {"local", "pinecone"}:
            raise ValueError("EMOTION_CONTEXT_BACKEND must be 'local' or 'pinecone'.")

        self.index = None
        if self.backend == "pinecone":
            from pinecone import os

import Pinecone

            api_key = os.getenv("PINECONE_API_KEY", "").strip()
            index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
            self.namespace = os.getenv("PINECONE_NAMESPACE", "emotion-context").strip()
            if not api_key or not index_name or not self.namespace:
                raise RuntimeError(
                    "Pinecone backend requires PINECONE_API_KEY, "
                    "PINECONE_INDEX_NAME, and PINECONE_NAMESPACE."
                )
            self.index = Pinecone(api_key=api_key).Index(index_name)

        # Psychology-backed context examples
        # In a real RAG pipeline, these would be chunked embeddings.
        self.psychology_database = {
            "happy": (
                "Happiness is associated with dopamine release, social bonding, "
                "and increased cognitive flexibility. People experiencing happiness "
                "tend to display open body language and increased eye contact."
            ),
            "sad": (
                "Sadness often arises from loss, disappointment, or emotional fatigue. "
                "It may lead to withdrawal, lowered mood, reduced energy, and slower speech."
            ),
            "angry": (
                "Anger is connected to perceived threats or injustice. Physiological signs "
                "include increased heart rate, tense muscles, and direct gaze."
            ),
            "fear": (
                "Fear activates the amygdala, triggering fight-or-flight responses. "
                "Common behaviors include avoidance, widened eyes, and defensive posture."
            ),
            "neutral": (
                "Neutral expressions indicate a baseline emotional state. The individual may "
                "be processing information, maintaining composure, or simply at rest."
            ),
            "surprise": (
                "Surprise is linked to sudden unexpected stimuli. Behavioral cues include "
                "raised eyebrows, wide eyes, and a brief pause in movement."
            ),
            "disgust": (
                "Disgust often emerges as a reaction to unpleasant or morally "
                "objectionable stimuli. "
                "It is associated with nose wrinkling, eye narrowing, and head turning."
            ),
        }

    def retrieve(self, emotions: list[str]) -> str:
        """
        Retrieve psychology context for a list of predicted emotions.

        Parameters
        ----------
        emotions : list of str
            Example: ['sad', 'fear']

        Returns
        -------
        str : Combined explanatory context
        """

        if self.backend == "pinecone":
            from openai import OpenAI

            query = " ".join(emotions)
            vector = OpenAI().embeddings.create(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
                input=query,
            ).data[0].embedding
            response = self.index.query(
                vector=vector,
                top_k=min(len(emotions) or 1, 5),
                include_metadata=True,
                namespace=self.namespace,
            )
            return " ".join(
                match.metadata.get("text", "")
                for match in response.matches
                if match.metadata and match.metadata.get("text")
            )

        context_chunks = []

        for emotion in emotions:
            emotion = emotion.lower().strip()
            if emotion in self.psychology_database:
                context_chunks.append(self.psychology_database[emotion])
            else:
                context_chunks.append(f"No psychology context available for '{emotion}'.")

        # Merge all context into one unified RAG context block
        return " ".join(context_chunks)
