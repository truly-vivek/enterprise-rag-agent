import pytest

from app.main import KnowledgeBase


def test_chunks_include_expected_overlap():
    chunks = KnowledgeBase.split_text("abcdefghij", size=6, overlap=2)
    assert chunks == ["abcdef", "efghij"]


def test_empty_text_returns_no_chunks():
    assert KnowledgeBase.split_text("   ", size=10, overlap=2) == []


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase.split_text("text", size=5, overlap=5)


def test_zero_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase.split_text("text", size=0, overlap=0)


def test_document_text_extraction():
    assert KnowledgeBase.extract_text("notes.txt", b"hello world") == "hello world"


def test_unsupported_file_type_is_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase.extract_text("archive.zip", b"content")
