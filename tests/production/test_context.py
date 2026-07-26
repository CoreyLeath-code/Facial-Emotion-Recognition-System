import sys
from types import ModuleType, SimpleNamespace

import pytest

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


def test_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("EMOTION_CONTEXT_BACKEND", "remote")
    with pytest.raises(ValueError, match="must be 'local' or 'pinecone'"):
        EmotionContextRetriever()


def test_pinecone_requires_configuration(monkeypatch):
    pinecone_module = ModuleType("pinecone")
    pinecone_module.Pinecone = object
    monkeypatch.setitem(sys.modules, "pinecone", pinecone_module)
    monkeypatch.setenv("EMOTION_CONTEXT_BACKEND", "pinecone")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_NAME", raising=False)

    with pytest.raises(RuntimeError, match="Pinecone backend requires"):
        EmotionContextRetriever()


def test_pinecone_retrieval_uses_embedding_and_namespace(monkeypatch):
    class FakeIndex:
        def query(self, **kwargs):
            assert kwargs["namespace"] == "test-emotions"
            assert kwargs["top_k"] == 2
            return SimpleNamespace(
                matches=[
                    SimpleNamespace(metadata={"text": "happy context"}),
                    SimpleNamespace(metadata={"text": "calm context"}),
                    SimpleNamespace(metadata=None),
                ]
            )

    class FakePinecone:
        def __init__(self, api_key):
            assert api_key == "pinecone-test-key"

        def Index(self, index_name):
            assert index_name == "emotion-index"
            return FakeIndex()

    class FakeOpenAI:
        def __init__(self):
            self.embeddings = SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    data=[SimpleNamespace(embedding=[0.1, 0.2])]
                )
            )

    pinecone_module = ModuleType("pinecone")
    pinecone_module.Pinecone = FakePinecone
    openai_module = ModuleType("openai")
    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "pinecone", pinecone_module)
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    monkeypatch.setenv("EMOTION_CONTEXT_BACKEND", "pinecone")
    monkeypatch.setenv("PINECONE_API_KEY", "pinecone-test-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "emotion-index")
    monkeypatch.setenv("PINECONE_NAMESPACE", "test-emotions")

    result = EmotionContextRetriever().retrieve(["happy", "calm"])

    assert result == "happy context calm context"
