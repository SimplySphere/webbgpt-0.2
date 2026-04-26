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


def _domain_source() -> DataSourceConfig:
    return DataSourceConfig(
        name="catalog_expanded_corpus",
        format="text",
        quality_filter=True,
        quality_filter_mode="domain_lm",
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


def test_domain_lm_filter_removes_source_section_scaffolding_from_kept_text():
    text = (
        "Source: course_catalog_2025_26.html. Section: Course Planning. "
        "Students should plan their programs for the full school year, with subsequent years of study "
        "and college in mind."
    )

    result = clean_document(DocumentRecord(text=text, source="catalog"), DataConfig(min_document_chars=1), _domain_source())

    assert result.record is not None
    assert result.dropped_reason is None
    assert result.record.text == (
        "Course Planning. Students should plan their programs for the full school year, "
        "with subsequent years of study and college in mind."
    )


def test_domain_lm_filter_drops_table_of_contents_fragments():
    text = "Source: handbook.txt. Section: Contents. Honor............................................................ 10"

    result = clean_document(DocumentRecord(text=text, source="handbook"), DataConfig(min_document_chars=1), _domain_source())

    assert result.record is None
    assert result.dropped_reason == "domain_lm_table_of_contents"


def test_domain_lm_filter_drops_short_metadata_fragments_after_cleanup():
    text = "Source: college_guidance.html. Section: Top 40 Colleges Webb Students Matriculate To Most:. Babson College"

    result = clean_document(DocumentRecord(text=text, source="advising"), DataConfig(min_document_chars=1), _domain_source())

    assert result.record is None
    assert result.dropped_reason == "domain_lm_list_fragment"


def test_domain_lm_filter_drops_structured_source_junk():
    text = (
        "College Guidance Team Photo of Hector Martinez Dean of College Guidance Photo of Rhemi Abrams-Fuller "
        "Associate Dean of College Guidance. This line is long enough to pass basic length checks, "
        "but it is media chrome rather than coherent prose for language modeling."
    )

    result = clean_document(DocumentRecord(text=text, source="advising"), DataConfig(min_document_chars=1), _domain_source())

    assert result.record is None
    assert result.dropped_reason == "domain_lm_structured_source_junk"
