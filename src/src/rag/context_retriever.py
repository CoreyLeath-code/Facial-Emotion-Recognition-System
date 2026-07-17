class EmotionContextRetriever:
    """
    Lightweight RAG retriever for psychological explanations tied to emotion classes.
    This simulates vector retrieval for a GitHub portfolio project but can be upgraded
    to FAISS/Pinecone later.
    """

    def __init__(self) -> None:
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

        context_chunks = []

        for emotion in emotions:
            emotion = emotion.lower().strip()
            if emotion in self.psychology_database:
                context_chunks.append(self.psychology_database[emotion])
            else:
                context_chunks.append(f"No psychology context available for '{emotion}'.")

        # Merge all context into one unified RAG context block
        return " ".join(context_chunks)
