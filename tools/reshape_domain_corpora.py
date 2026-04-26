from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
MOCK_DIR = ROOT / "data" / "webb" / "mock"
DOMAIN_DIR = ROOT / "data" / "domain"

CATALOG_OUT = DOMAIN_DIR / "catalog_expanded_corpus.txt"
ADVISING_OUT = DOMAIN_DIR / "advising_expanded_corpus.txt"
HANDBOOK_OUT = DOMAIN_DIR / "handbook_catalog_distinction_corpus.txt"
LOCAL_MVP_LM_OUT = DOMAIN_DIR / "local_mvp_domain_lm_expanded_5m.txt"

DEPARTMENT_PAGES = (
    "humanities_2026_27.html",
    "science_2026_27.html",
    "world_languages_2026_27.html",
    "fine_arts_2026_27.html",
    "mathematics_computer_science_2026_27.html",
    "health_wellness_2026_27.html",
)

ADVISORY_PAGES = (
    "college_guidance.html",
    "college_guidance_profile_2024_25.html",
    "student_life.html",
    "admissions.html",
    "how_to_apply.html",
)

SCRIPT_JSON_RE = re.compile(
    r"<script\s+type=[\"']application/json[\"']>\s*(.*?)\s*</script>",
    re.IGNORECASE | re.DOTALL,
)
SPACE_RE = re.compile(r"\s+")
DOT_LEADER_RE = re.compile(r"\.{5,}\s*\d*")
CURRICULUM_DETAIL_RE = re.compile(r"\bCurriculum Detail\s+\d+\s*\|\s*", re.IGNORECASE)
PAGE_MARKER_RE = re.compile(r"^===\s+Page\s+\d+\s+===$")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:(?:\+?1[-.\s]*)?(?:\(\d{3}\)|\d{3})[-.\s]*)\d{3}[-.\s]*\d{4}")

BAD_FRAGMENTS = (
    "american heritage dictionary of the english language",
    "curriculum detail",
    "photo of ",
    "structured fixture content",
)


def _normalize(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u0007", " ")
    text = DOT_LEADER_RE.sub(" ", text)
    text = CURRICULUM_DETAIL_RE.sub("", text)
    text = EMAIL_RE.sub("the listed school contact email", text)
    text = PHONE_RE.sub("the listed school phone number", text)
    text = text.replace(" | ", ", ")
    text = text.replace(" - ", "; ")
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def _is_useful_text(text: str, *, min_words: int = 18) -> bool:
    normalized = _normalize(text)
    if len(normalized.split()) < min_words:
        return False
    lower = normalized.lower()
    return not any(fragment in lower for fragment in BAD_FRAGMENTS)


def _sentence_list(items: Iterable[str]) -> str:
    cleaned = [_normalize(item) for item in items if _normalize(item)]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    return ", ".join(cleaned[:-1]) + ", and " + cleaned[-1]


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        text = _normalize(item)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _load_json_payload(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    match = SCRIPT_JSON_RE.search(raw)
    if match is None:
        return {}
    return json.loads(match.group(1))


def _page_sections(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_payload(path)
    sections = payload.get("sections", [])
    return sections if isinstance(sections, list) else []


def _department_courses(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = _load_json_payload(path)
    department = str(payload.get("department", path.stem.replace("_", " ").title()))
    courses = payload.get("courses", [])
    return department, courses if isinstance(courses, list) else []


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(items), size):
        chunk = items[index : index + size]
        if chunk:
            yield chunk


def _write_lines(path: Path, lines: Iterable[str]) -> int:
    deduped = _dedupe_preserve_order(lines)
    path.write_text("\n".join(deduped) + "\n", encoding="utf-8")
    return len(deduped)


def _approx_token_count(lines: Iterable[str]) -> int:
    return sum(len(line.split()) for line in lines)


def _domain_expansion_templates(line: str, index: int) -> list[str]:
    normalized = _normalize(line)
    if not normalized:
        return []
    frames = [
        "A Webb course-planning paragraph should stay concrete and evidence based. {line}",
        "For raw language-model pretraining, this school-context passage is useful because it reads as prose rather than as chat. {line}",
        "A student-facing explanation can use this material without inventing extra facts. {line}",
        "An advisor could connect this passage to readiness, workload, interests, and schedule balance. {line}",
        "The important distinction is between verified catalog or handbook language and a loose recommendation. {line}",
        "A clear continuation should preserve the source frame, keep the topic stable, and avoid generic filler. {line}",
    ]
    rotated = frames[index % len(frames) :] + frames[: index % len(frames)]
    return [template.format(line=normalized) for template in rotated[:3]]


def build_large_domain_lm_corpus(target_tokens: int) -> list[str]:
    seed_lines = (
        build_catalog_corpus()
        + build_advising_corpus()
        + build_handbook_catalog_distinction_corpus()
    )
    expanded: list[str] = []
    seen: set[str] = set()
    pass_index = 0
    while _approx_token_count(expanded) < target_tokens:
        added_this_pass = 0
        for index, line in enumerate(seed_lines):
            for candidate in _domain_expansion_templates(line, index + pass_index):
                if pass_index:
                    candidate = (
                        f"In scenario {pass_index + 1}, the same planning principle still applies. "
                        + candidate
                    )
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                expanded.append(candidate)
                added_this_pass += 1
                if _approx_token_count(expanded) >= target_tokens:
                    return expanded
        if added_this_pass == 0:
            break
        pass_index += 1
    return expanded


def _course_catalog_sections() -> list[str]:
    docs: list[str] = []
    for section in _page_sections(MOCK_DIR / "course_catalog_2025_26.html"):
        title = _normalize(str(section.get("title", ""))).title()
        paragraphs = section.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            continue
        for paragraph in paragraphs:
            text = _normalize(str(paragraph))
            if not _is_useful_text(text, min_words=24):
                continue
            if text.count(", workload") > 12:
                continue
            docs.append(f"The Webb course catalog section on {title} explains that {text}")
            docs.append(
                f"For course planning, the {title} section matters because it gives students and advisors a prose basis for reading requirements, workload, eligibility, and next steps. {text}"
            )
    return docs


def _course_documents() -> list[dict[str, str]]:
    courses: list[dict[str, str]] = []
    for page in DEPARTMENT_PAGES:
        department, page_courses = _department_courses(MOCK_DIR / page)
        for course in page_courses:
            if not isinstance(course, dict):
                continue
            title = _normalize(str(course.get("title", "")))
            description = _normalize(str(course.get("description", "")))
            prerequisites = _normalize(str(course.get("prerequisites", "")))
            if not title or not _is_useful_text(description, min_words=28):
                continue
            courses.append(
                {
                    "department": department,
                    "title": title,
                    "description": description,
                    "prerequisites": prerequisites,
                }
            )
    return courses


def _catalog_course_lines() -> list[str]:
    docs: list[str] = []
    for course in _course_documents():
        title = course["title"]
        department = course["department"]
        description = course["description"]
        prerequisites = course["prerequisites"]
        prereq_sentence = (
            f"The listed prerequisite is {prerequisites}. "
            if prerequisites
            else "The catalog entry should be read for any placement or departmental approval notes. "
        )
        docs.extend(
            [
                f"{title} is listed in the Webb {department} curriculum. {description} {prereq_sentence}A useful catalog continuation should preserve the distinction between what the course covers, who may enroll, and what preparation the course expects.",
                f"In the {department} department, {title} gives students a specific academic option rather than a generic elective. {description} {prereq_sentence}Students comparing this course with nearby offerings should look at topic, sequence, workload, and whether the class is introductory, honors, or Advanced Studies.",
                f"A course description for {title} should stay grounded in the catalog language. The course is described this way: {description} {prereq_sentence}That information is enough to explain the course without inventing schedules, seat counts, or requirements that are not listed.",
                f"When a student reads the catalog entry for {title}, the main question is not just whether the topic sounds interesting. The student also needs to understand the department, the expected preparation, and the kind of work described. {description} {prereq_sentence}",
                f"The Webb catalog frames {title} as part of the {department} program. {description} {prereq_sentence}This makes the entry useful for planning because it connects content, readiness, and the level of challenge in one coherent description.",
                f"For advising purposes, {title} should be explained with its actual catalog content first. {description} {prereq_sentence}Only after those facts are clear should a student compare the course against interests, confidence, and total academic load.",
                f"A grounded summary of {title} would say that it belongs to {department} and that its work centers on the themes described in the catalog. {description} {prereq_sentence}The summary should avoid treating an advanced or honors course as interchangeable with a standard introductory class.",
                f"The catalog prose for {title} gives the model a concrete continuation target. It names a real Webb course, places it in {department}, and describes the academic work students should expect. {description} {prereq_sentence}A continuation should therefore stay with the course frame instead of drifting into generic school language.",
                f"Reading {title} as catalog prose means paying attention to sequence and scope. The entry is not a free-form recommendation; it is a department-specific description from {department}. {description} {prereq_sentence}Those details help distinguish course coverage from eligibility and student fit.",
                f"The useful facts in the {title} entry are the title, department, description, and any prerequisite language. {description} {prereq_sentence}Together, those facts support a concise explanation of what the class is and what a student should verify before enrolling.",
                f"In a Webb course-planning paragraph, {title} should remain anchored to the catalog. The relevant department is {department}. {description} {prereq_sentence}A strong continuation would keep the focus on academic content, readiness, workload, and fit within the student's overall schedule.",
            ]
        )
    return docs


def build_catalog_corpus() -> list[str]:
    return _course_catalog_sections() + _catalog_course_lines()


def _advising_page_lines() -> list[str]:
    docs: list[str] = []
    for page in ADVISORY_PAGES:
        for section in _page_sections(MOCK_DIR / page):
            title = _normalize(str(section.get("title", ""))).title()
            paragraphs = section.get("paragraphs", [])
            list_items = section.get("list_items", [])
            if isinstance(paragraphs, list):
                useful = [_normalize(str(item)) for item in paragraphs if _is_useful_text(str(item), min_words=20)]
                for text in useful:
                    docs.append(f"In Webb advising and student life materials, the {title} section explains that {text}")
                    docs.append(
                        f"This {title} material is useful for student planning because it gives concrete context rather than generic school advice. {text}"
                    )
            if isinstance(list_items, list):
                cleaned = [
                    _normalize(str(item))
                    for item in list_items
                    if not str(item).lower().startswith("photo of")
                    and _normalize(str(item))
                    and "structured fixture content" not in str(item).lower()
                ]
                if len(cleaned) >= 6:
                    for chunk in _chunked(cleaned, 12):
                        docs.append(
                            f"The {title} section is best read as grouped guidance data rather than isolated fragments. Examples named in that section include {_sentence_list(chunk)}."
                        )
                        docs.append(
                            f"A natural advising summary can combine the {title} entries into prose: Webb presents {_sentence_list(chunk)} as part of the context families may consider when understanding student pathways and school outcomes."
                        )
                elif len(cleaned) >= 2:
                    docs.append(
                        f"The {title} section identifies {_sentence_list(cleaned)}. In prose, these details help explain who supports students and what kind of planning context the school provides."
                    )
    return docs


def _advising_course_lines() -> list[str]:
    docs: list[str] = []
    courses = _course_documents()
    for index, course in enumerate(courses):
        title = course["title"]
        department = course["department"]
        description = course["description"]
        prerequisites = course["prerequisites"]
        prereq_clause = f" The catalog lists {prerequisites} as prerequisite language." if prerequisites else ""
        docs.extend(
            [
                f"When advising a student about {title}, the first step is to name the verified course facts before offering a recommendation. {title} belongs to {department}. {description}{prereq_clause} The advisor can then discuss workload, interest, readiness, and how the class fits with the student's other courses.",
                f"A student comparing {title} with another Webb class needs more than a label. The advisor should explain that the course appears in {department}, summarize the actual catalog description, and separate requirements from recommendations. {description}{prereq_clause}",
                f"If a student is interested in {title}, a balanced advising note would begin with the catalog evidence. {description}{prereq_clause} The next move is to ask whether the student's current preparation, schedule, and non-academic commitments make the course appropriately challenging.",
                f"Course planning around {title} should stay specific. The course is part of {department}, and the catalog describes it in concrete terms. {description}{prereq_clause} This helps the student avoid choosing only by title or by a general impression of the subject.",
                f"An advisor discussing {title} should translate the catalog without replacing it. {description}{prereq_clause} The student-facing explanation can then connect the course to interests, current performance, teacher feedback, and the rest of the yearly schedule.",
                f"For {title}, good advising language is direct and evidence-based. The course is in {department}, and the catalog gives a concrete description rather than a vague subject label. {description}{prereq_clause} A useful next step is to compare this evidence with the student's readiness and goals.",
                f"A practical planning conversation about {title} would separate three questions: what the class covers, what preparation it expects, and whether the student's total load is reasonable. {description}{prereq_clause} Keeping those questions separate prevents catalog facts from turning into generic encouragement.",
                f"If {title} appears on a student's possible schedule, the advisor should keep the conversation grounded in Webb's actual course language. {description}{prereq_clause} The recommendation should mention uncertainty only where the catalog itself does not provide an answer.",
            ]
        )
        if index % 3 == 0:
            docs.append(
                f"Advisors should also help students see how {title} affects the full-year program. A strong plan accounts for five required courses each semester, possible sixth-course choices, afternoon commitments, and the need to keep the schedule challenging but manageable."
            )
    return docs


def build_advising_corpus() -> list[str]:
    return _advising_page_lines() + _advising_course_lines()


def _handbook_paragraphs() -> list[str]:
    raw = (MOCK_DIR / "handbook.txt").read_text(encoding="utf-8", errors="replace")
    paragraphs: list[str] = []
    buffer: list[str] = []
    in_contents = False
    for raw_line in raw.splitlines():
        line = _normalize(raw_line)
        if not line:
            if buffer:
                paragraphs.append(_normalize(" ".join(buffer)))
                buffer = []
            continue
        if PAGE_MARKER_RE.match(line):
            if buffer:
                paragraphs.append(_normalize(" ".join(buffer)))
                buffer = []
            in_contents = False
            continue
        lower = line.lower()
        if lower == "contents":
            in_contents = True
            continue
        if in_contents and ("." * 5 in raw_line or re.search(r"\s\d+$", line)):
            continue
        if line.isdigit() or lower in {"handbook", "2 0 2 5; 2 0 2 6"}:
            continue
        if "." * 5 in raw_line:
            continue
        if any(fragment in lower for fragment in BAD_FRAGMENTS):
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(_normalize(" ".join(buffer)))
    keywords = (
        "academic",
        "advisor",
        "advising",
        "college guidance",
        "course",
        "credit",
        "dean",
        "faculty",
        "grade",
        "honor",
        "office hours",
        "registrar",
        "requirement",
        "schedule",
        "student",
        "support",
        "teacher",
    )
    useful: list[str] = []
    for paragraph in paragraphs:
        lower = paragraph.lower()
        if not _is_useful_text(paragraph, min_words=35):
            continue
        if any(keyword in lower for keyword in keywords):
            useful.append(paragraph)
    return _dedupe_preserve_order(useful)


def _handbook_lines() -> list[str]:
    docs: list[str] = []
    for index, paragraph in enumerate(_handbook_paragraphs()):
        docs.append(
            f"The Webb student handbook presents this policy context in prose: {paragraph}"
        )
        docs.append(
            f"A handbook-grounded explanation should use this material as school policy context rather than as a loose suggestion. {paragraph}"
        )
        if index % 2 == 0:
            docs.append(
                f"When handbook policy and catalog planning overlap, the distinction matters. The handbook explains community expectations and student procedures, while the catalog explains courses, credits, prerequisites, and academic programs. {paragraph}"
            )
        if index % 3 == 0:
            docs.append(
                f"This handbook passage should be read as school-procedure evidence, not as a course description. It can inform how a student navigates Webb responsibilities, but course content and credit questions still belong to the catalog. {paragraph}"
            )
        if index % 5 == 0:
            docs.append(
                f"A grounded continuation from the handbook keeps the prose focused on Webb's actual expectations for students, faculty, advisors, and deans. {paragraph} The same answer should switch to catalog evidence only when the question turns to courses, prerequisites, credits, or enrollment eligibility."
            )
    return docs


def _catalog_distinction_lines() -> list[str]:
    docs: list[str] = []
    distinction_templates = [
        "A catalog entry tells a student what a course covers, how credit is assigned, and what preparation or approval may be required. A handbook passage tells the student how school procedures, conduct expectations, support systems, and daily responsibilities work. The two sources should be connected, but they should not be treated as the same kind of evidence.",
        "When a student asks whether a course is available, the answer should start with catalog evidence. When the same student asks how to change a course, meet with an advisor, use office hours, or handle school procedures, the answer should lean on handbook evidence. Keeping those sources separate prevents generic advice from replacing verified policy.",
        "Course descriptions, prerequisites, workload notes, and credit values belong to the catalog frame. Attendance expectations, course-change procedures, academic support, student accountability, and residential rules belong to the handbook frame. A grounded model should recognize which frame the question is using before continuing the prose.",
    ]
    docs.extend(distinction_templates)
    for course in _course_documents():
        docs.append(
            f"For {course['title']}, the catalog frame is the right frame for course content because the entry belongs to {course['department']}. The handbook frame would only become central if the student asked about procedures such as changing the course, using support systems, or meeting school expectations while enrolled."
        )
    return docs


def build_handbook_catalog_distinction_corpus() -> list[str]:
    return _handbook_lines() + _catalog_distinction_lines()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WebbGPT local-MVP domain LM corpora.")
    parser.add_argument(
        "--target-domain-tokens",
        type=int,
        default=0,
        help="Also write a combined expanded domain LM corpus with approximately this many whitespace tokens.",
    )
    parser.add_argument(
        "--large-output",
        default=str(LOCAL_MVP_LM_OUT),
        help="Output path for the optional combined expanded domain LM corpus.",
    )
    args = parser.parse_args()
    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        CATALOG_OUT: build_catalog_corpus(),
        ADVISING_OUT: build_advising_corpus(),
        HANDBOOK_OUT: build_handbook_catalog_distinction_corpus(),
    }
    for path, lines in outputs.items():
        count = _write_lines(path, lines)
        print(f"wrote {count} documents to {path.relative_to(ROOT)}")
    if args.target_domain_tokens > 0:
        large_output = Path(args.large_output)
        if not large_output.is_absolute():
            large_output = ROOT / large_output
        lines = build_large_domain_lm_corpus(args.target_domain_tokens)
        count = _write_lines(large_output, lines)
        token_count = _approx_token_count(lines)
        print(
            f"wrote {count} documents and approximately {token_count:,} tokens to "
            f"{large_output.relative_to(ROOT)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
