from config import DataConfig, DataSourceConfig
from data.preprocess import DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES
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
        name="catalog_domain_fixture",
        format="text",
        quality_filter=True,
        quality_filter_mode="domain_lm",
        pii_scrub=False,
    )


def _curated_source() -> DataSourceConfig:
    return DataSourceConfig(
        name="fineweb",
        format="text",
        quality_filter=True,
        quality_filter_mode="curated_lm",
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


def test_domain_lm_filter_drops_synthetic_training_scaffolds():
    for phrase in DOMAIN_LM_SYNTHETIC_SCAFFOLD_PHRASES:
        text = (
            "AdvSt Literature and The Machine is a Humanities course that asks students to read, "
            "write, and think about technology in relation to literary history. "
            f"This generated row accidentally includes the synthetic phrase {phrase}. "
            "The rest of the paragraph is long enough that it would otherwise look like domain prose."
        )

        result = clean_document(
            DocumentRecord(text=text, source="catalog"),
            DataConfig(min_document_chars=1),
            _domain_source(),
        )

        assert result.record is None
        assert result.dropped_reason == "domain_lm_synthetic_training_scaffold"


def test_broad_lm_filter_does_not_apply_domain_scaffold_reject_reason():
    text = (
        "A general article about machine learning may refer to the model while explaining how a system "
        "learns from examples. The paragraph is ordinary broad prose, not Webb domain corpus expansion, "
        "and should not receive a domain-specific synthetic scaffold reject reason."
    )

    result = clean_document(
        DocumentRecord(text=text, source="fineweb"),
        DataConfig(min_document_chars=1),
        _broad_source(),
    )

    assert result.record is not None
    assert result.dropped_reason is None


def test_curated_lm_filter_keeps_clean_real_prose_shape():
    text = (
        "A clear explanatory passage usually gives readers a concrete setting before it names the larger idea. "
        "For example, a lesson about weather can begin with the way clouds gather over a valley in the afternoon. "
        "The writer can then explain how warm air rises, cools, and forms visible droplets. "
        "Because each sentence adds one piece of the explanation, the paragraph keeps its topic without sounding like a list. "
        "That kind of ordinary prose is useful for language-model pretraining because it teaches continuation, pacing, and context."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is not None
    assert result.dropped_reason is None


def test_curated_lm_filter_drops_product_commercial_pages():
    text = (
        "This review compares the best price for a portable desk lamp and explains why shoppers should buy now. "
        "The product page lists customer reviews, ratings, warranty details, and free shipping offers. "
        "A discount coupon appears next to the cart, and the store repeats the sale message several times. "
        "Although the paragraph has sentences, the commercial language dominates the page. "
        "It should be rejected from a curated prose recipe because the continuation would teach shopping drift. "
        "A scalable filter can still keep useful how-to writing while excluding pages organized around checkout intent."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_product_commercial_dense"


def test_curated_lm_filter_does_not_count_product_terms_inside_words():
    text = (
        "A workshop on production planning can still be ordinary explanatory prose. "
        "The passage describes how a team makes a productive schedule, tracks materials, and reviews results. "
        "It does not ask readers to buy anything, compare prices, open a cart, or read customer ratings. "
        "Because the terms appear inside larger words, they should not be counted as commercial page artifacts. "
        "The document has enough connected sentences to remain useful for broad pretraining."
        "A reader could continue the explanation by describing the planning problem, the constraints, and the evidence. "
        "Those details make it connected practical prose rather than a shopping page."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is not None
    assert result.dropped_reason is None


def test_curated_lm_filter_drops_dictionary_fragments():
    text = (
        "The entry begins with a noun definition, a plural form, pronunciation notes, and a short etymology. "
        "It lists synonyms and antonyms before adding a dictionary example that does not develop an idea. "
        "The encyclopedia fragment is formulaic because it catalogs word origin and part of speech information. "
        "A paragraph like this may be accurate, however it is too close to a lookup page. "
        "The curated pretraining recipe should prefer connected prose over thesaurus and glossary fragments. "
        "The issue is not that reference material is useless, but that this shape rarely teaches paragraph continuation."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_dictionary_or_encyclopedia_fragment"


def test_curated_lm_filter_drops_page_instruction_boilerplate():
    text = (
        "Simply begin typing or use the editing tools above to add to this article. "
        "Once you are finished and click submit, your changes will be reviewed by editors. "
        "Send the link below via email so invited audience members can follow the remote presentation. "
        "The surrounding paragraph has sentence boundaries and enough words to look superficially usable. "
        "However, this is page-instruction boilerplate rather than natural explanatory prose for pretraining. "
        "The curated recipe should remove it before repeated page scaffolds become a common continuation."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_page_boilerplate"


def test_curated_lm_filter_drops_familysearch_edit_page_boilerplate():
    text = (
        "Hungary Church RecordsEdit This Page From FamilySearch Wiki Back to Hungary Page. "
        "Church registers refer to records of births, marriages, deaths, and burials recorded by churches. "
        "The page includes historical details about record keeping, parish practice, and civil archives. "
        "However, the source header is scrape furniture from an editable wiki rather than a clean article opening. "
        "The curated recipe should reject this specific page chrome without rejecting ordinary writing about families."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_familysearch_wiki_boilerplate"


def test_curated_lm_filter_drops_html_archive_boilerplate():
    text = (
        "The following HTML text is provided to enhance online readability. "
        "Many aspects of typography translate only awkwardly to HTML, so the page image is described as authoritative. "
        "The remaining passage contains several explanatory sentences about a case study and its surrounding concepts. "
        "Even when the topic is coherent, this header is archival page boilerplate and should not be repeated in pretraining. "
        "The filter should remove the page before the source wrapper becomes a learned continuation."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_html_archive_boilerplate"


def test_curated_lm_filter_drops_science_fair_encyclopedia_residue():
    text = (
        "Science Fair Project Encyclopedia Misapplied to Greek mythology begins with a reference-style heading. "
        "The following sentences define a demigod, redirect readers toward another concept, and summarize a lookup entry. "
        "Although the page has readable prose, the source frame is a formulaic project encyclopedia fragment. "
        "A curated local pretraining corpus should prefer ordinary connected explanation over repeated index residue. "
        "This narrow phrase catches the audited source without banning normal science writing."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_science_fair_encyclopedia"


def test_curated_lm_filter_drops_cookie_video_rating_widgets():
    text = (
        "Please activate cookies in your browser so this video page can load the player correctly. "
        "Rating is available when the video has been rented, and visitors may sign in to add this video to a playlist. "
        "The surrounding text describes a lecture, a speaker, and a public discussion in complete sentences. "
        "Those sentences are still wrapped in interaction widgets rather than paragraph prose. "
        "The curated recipe should remove this exact page pattern without banning ordinary references to baked cookies."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_cookie_video_widget_boilerplate"


def test_curated_lm_filter_keeps_natural_cookie_prose():
    text = (
        "A baker can test a cookie recipe by changing one ingredient at a time and writing down the result. "
        "For example, more brown sugar usually makes the center softer, while a longer bake makes the edge crisp. "
        "The useful part of the experiment is not the dessert itself but the habit of observing cause and effect. "
        "Because the paragraph stays on one practical topic, it is normal everyday prose rather than a web widget. "
        "A reader could continue by comparing flour types, oven temperature, and cooling time."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is not None
    assert result.dropped_reason is None


def test_curated_lm_filter_drops_newsletter_source_taglines():
    text = (
        "The latest news from academia, regulators research labs and other things of interest appeared above the article. "
        "The story then described a laboratory result, a public agency response, and several implications for future work. "
        "Although the article body contains real sentences, the repeated source tagline was one of the audited attractors. "
        "A small language model can learn that opener too easily when repeated documents are enabled. "
        "The curated selector should reject the narrow tagline rather than general science prose."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_newsletter_or_source_tagline"


def test_curated_lm_filter_drops_repeated_lookup_and_source_ctas():
    cases = [
        (
            "One of the mysteries of the English language finally explained before a list of dictionary examples. "
            "The following passage has several sentences, quotations, and explanatory-looking fragments. "
            "However, the audited source repeats the same lookup-page opener across unrelated entries. "
            "That makes it a poor fit for a small raw language-model corpus even if some definitions are readable. "
            "The curated selector should catch this exact repeated lookup frame.",
            "curated_lm_dictionary_or_encyclopedia_fragment",
        ),
        (
            "Find more Browneller relatives and grow your tree by exploring billions of historical records. "
            "Taken every decade since 1790, the census can tell a reader about a family name and its locations. "
            "The text has ordinary sentences, but the repeated genealogy call to action was a top n-gram after filtering. "
            "It should be treated as source boilerplate rather than as general explanatory prose. "
            "The rest of the paragraph is long enough to pass basic shape checks.",
            "curated_lm_newsletter_or_source_tagline",
        ),
        (
            "700 Journals and 15,000,000 Readers Each Journal is getting 25,000+ ReadersThis Readership is 10 times more "
            "when compared to other Subscription Journals. The article then begins to describe a research study. "
            "This promotional header is repeated source furniture rather than useful academic prose. "
            "The curated selector should reject it before the repeat cap gives it extra exposure. "
            "A clean journal article without the promotional wrapper can still be kept.",
            "curated_lm_newsletter_or_source_tagline",
        ),
        (
            "The leading eBooks store online for Kindle Fire, Apple, Android, Nook, Kobo, PC, Mac, and Sony devices "
            "appears before a short article about reading formats. The paragraph has sentence boundaries and enough words. "
            "Even so, the audited phrase is commercial store boilerplate, not ordinary prose. "
            "The curated selector should reject this exact product widget without banning normal book discussions. "
            "This prevents shopping continuations from appearing in unrelated prompts.",
            "curated_lm_product_commercial_dense",
        ),
    ]

    for text, reason in cases:
        result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())
        assert result.record is None
        assert result.dropped_reason == reason


def test_curated_lm_filter_drops_pipe_category_menu_chains():
    text = (
        "Individual differences | Methods | Statistics | Clinical | Educational | Industrial | Professional items | "
        "World psychology | The article then begins a paragraph about a research topic and its historical context. "
        "The body has enough sentences to look acceptable after normalization, but the opening is a category menu chain. "
        "That kind of pipe-separated source furniture was visible in the repeated n-gram audit. "
        "The curated selector should reject it before packing so the model does not learn menu continuations."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_category_menu_boilerplate"


def test_curated_lm_filter_drops_question_list_seo_fragments():
    text = (
        "How to write essay my school? What is 1 pages essay double spaced? "
        "Which essay format generator should a student use? Why do quizlet chapter words appear in the page? "
        "What are the top research paper topics good or bad? How many pages best essay writing service online gov? "
        "The source contains enough words and punctuation to look like prose, but it is really a question list. "
        "The curated selector should drop this SEO-shaped fragment before it becomes a continuation pattern."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_question_list_or_seo_fragments"


def test_curated_lm_filter_drops_medical_body_dense_pages():
    text = (
        "The health article discusses body pain, blood symptoms, infection, disease, and treatment in children. "
        "It mentions a doctor, diabetes, diet, viral spread, virus exposure, and medical advice in one short section. "
        "Because these terms dominate the prose, the passage would bias a tiny local run toward body-health continuation. "
        "The filter should reject it even though it has normal sentence boundaries and a coherent topic. "
        "The local MVP recipe is trying to reduce this drift in everyday practical prompts. "
        "A later specialist corpus can handle such topics deliberately instead of letting them leak into every prompt."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_medical_body_dense"


def test_curated_lm_filter_drops_repeated_ngram_pages():
    repeated = "the river bends around the old stone bridge"
    text = (
        f"{repeated}, and the town watches the water move slowly at dusk. "
        f"{repeated}, and the town watches the water move slowly at dusk. "
        f"{repeated}, and the town watches the water move slowly at dusk. "
        f"{repeated}, and the town watches the water move slowly at dusk. "
        "A final sentence tries to provide context, however the paragraph is mostly repeated scaffolding. "
        "The surrounding details are long enough for the normal document-length gate, but they do not fix the repeated span."
    )

    result = clean_document(DocumentRecord(text=text, source="fineweb"), DataConfig(), _curated_source())

    assert result.record is None
    assert result.dropped_reason == "curated_lm_repeated_ngram_heavy"
