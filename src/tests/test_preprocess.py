from config import DataConfig, DataSourceConfig
from data.preprocess import clean_document
from data.schemas import DocumentRecord


def _broad_source() -> DataSourceConfig:
    return DataSourceConfig(
        name="fineweb",
        format="text",
        quality_filter=True,
        quality_filter_mode="broad_lm",
        pii_scrub=False,
    )


def test_broad_lm_filter_drops_url_heavy_documents():
    text = (
        "This page collects classroom resources and short public explanations for students. "
        "The surrounding prose is mostly readable, but the source is dominated by outbound links. "
        "Visit https://example.com/a and https://example.org/b and www.example.net/c for more. "
        "The remaining text is only a wrapper around links rather than a clean paragraph for LM training."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _broad_source())

    assert result.record is None
    assert result.dropped_reason == "broad_lm_url_heavy"


def test_broad_lm_filter_drops_navigation_boilerplate_documents():
    text = (
        "Skip to content\n"
        "Read more\n"
        "Subscribe\n"
        "Related articles\n"
        "Click here\n"
        "The document has a few readable words, but it is primarily navigation chrome and page furniture. "
        "It should not be treated as clean broad pretraining prose for the next local MVP run."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _broad_source())

    assert result.record is None
    assert result.dropped_reason == "broad_lm_navigation_boilerplate"


def test_broad_lm_filter_keeps_clean_paragraph_documents():
    text = (
        "Students often make stronger progress when a lesson connects a concrete example to a general idea. "
        "A teacher can introduce the situation, name the question, and then show how evidence changes the answer. "
        "That structure gives readers enough context to follow the point without relying on menus, link lists, "
        "or page metadata from a scraped website."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _broad_source())

    assert result.record is not None
    assert result.dropped_reason is None
    assert result.record.text.startswith("Students often make stronger progress")
