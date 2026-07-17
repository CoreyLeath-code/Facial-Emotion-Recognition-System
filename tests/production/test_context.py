from src.src.rag.context_retriever import EmotionContextRetriever


def test_retrieves_known_emotion_case_insensitively():
    result = EmotionContextRetriever().retrieve([" Happy "])
    assert "Happiness" in result


def test_unknown_emotion_is_explicit():
    result = EmotionContextRetriever().retrieve(["confused"])
    assert "No psychology context" in result


def test_multiple_contexts_are_combined():
    result = EmotionContextRetriever().retrieve(["sad", "fear"])
    assert "Sadness" in result and "Fear" in result
