"""Prefect pipeline for extracting and structuring CV/profile data from PDF files using pdfplumber."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber
from prefect import flow, get_run_logger, task

logger = logging.getLogger(__name__)


def _get_active_logger() -> Any:
    """Retrieve the Prefect run logger or fallback to the module logger.

    Returns:
        The active logger instance.
    """
    try:
        return get_run_logger()
    except Exception:
        return logger


@task(name="extract-pdf-content")
def extract_pdf_content(pdf_path: str | Path) -> dict[str, Any]:
    """Extract metadata, per-page text, and sidebar content from a PDF using pdfplumber.

    Args:
        pdf_path: Path to the target PDF file.

    Returns:
        Dictionary containing metadata, pages, raw_text, sidebar_text, and main_text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
    """
    log = _get_active_logger()
    target_path = Path(pdf_path)
    if not target_path.exists():
        msg = f"PDF file not found at: {target_path}"
        raise FileNotFoundError(msg)

    log.info("Opening PDF file with pdfplumber: %s", target_path)
    pages: list[dict[str, Any]] = []
    full_text_chunks: list[str] = []
    sidebar_chunks: list[str] = []
    main_chunks: list[str] = []

    with pdfplumber.open(target_path) as pdf:
        metadata = {str(k): str(v) for k, v in (pdf.metadata or {}).items()}

        for index, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            pages.append({"page_number": index + 1, "text": page_text})
            full_text_chunks.append(page_text)

            # LinkedIn PDF export has a left sidebar on the first page (~x0 < 200)
            if index == 0:
                sidebar_crop = page.crop((0, 0, 200, page.height))
                main_crop = page.crop((200, 0, page.width, page.height))
                sidebar_chunks.append(sidebar_crop.extract_text() or "")
                main_chunks.append(main_crop.extract_text() or "")
            else:
                main_chunks.append(page_text)

    raw_text = "\n".join(full_text_chunks)
    sidebar_text = "\n".join(sidebar_chunks)
    main_text = "\n".join(main_chunks)

    log.info("Extracted %d pages from %s", len(pages), target_path.name)
    return {
        "metadata": metadata,
        "pages": pages,
        "raw_text": raw_text,
        "sidebar_text": sidebar_text,
        "main_text": main_text,
    }


def _parse_sidebar_sections(sidebar_text: str) -> dict[str, str]:
    """Parse distinct sections from the LinkedIn sidebar text.

    Args:
        sidebar_text: Text extracted from the first-page sidebar.

    Returns:
        Mapping of section names to their raw text content.
    """
    section_headers = [
        "Contato",
        "Principais competências",
        "Languages",
        "Certifications",
    ]
    indices: list[tuple[int, str]] = []
    for header in section_headers:
        idx = sidebar_text.find(header)
        if idx != -1:
            indices.append((idx, header))

    indices.sort(key=lambda x: x[0])
    parsed_sections: dict[str, str] = {}
    for i, (idx, name) in enumerate(indices):
        start = idx + len(name)
        end = indices[i + 1][0] if i + 1 < len(indices) else len(sidebar_text)
        parsed_sections[name] = sidebar_text[start:end].strip()

    return parsed_sections


def _extract_contact_info(
    contact_text: str,
    header_lines: list[str],
) -> dict[str, Any]:
    """Extract structured contact details such as email, LinkedIn, and location.

    Args:
        contact_text: Raw contact text from the sidebar.
        header_lines: Header lines from the main profile section.

    Returns:
        Dictionary with email, linkedin, location, and raw contact string.
    """
    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", contact_text)
    email = emails[0] if emails else None

    linkedin_match = re.search(
        r"(?:https?://)?(?:www\.)?linkedin\.com/in/\s*([^\s\(\)]+)",
        contact_text,
    )
    linkedin = (
        f"https://www.linkedin.com/in/{linkedin_match.group(1)}"
        if linkedin_match
        else None
    )

    location = None
    min_header_lines_for_location = 3
    if len(header_lines) >= min_header_lines_for_location:
        location = header_lines[-1].strip()

    return {
        "email": email,
        "linkedin": linkedin,
        "location": location,
        "raw": contact_text,
    }


def _parse_list_lines(text: str) -> list[str]:
    """Parse newline-delimited text into a clean list of strings.

    Args:
        text: Raw multi-line string.

    Returns:
        List of non-empty stripped strings with wrapped lines merged.
    """
    if not text:
        return []

    items: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if items and (line.startswith(("(", "and ")) or line[0].islower()):
            items[-1] = f"{items[-1]} {line}"
        else:
            items.append(line)
    return items


@task(name="parse-profile-structure")
def parse_profile_structure(extracted_content: dict[str, Any]) -> dict[str, Any]:
    """Parse extracted text into structured CV profile sections.

    Args:
        extracted_content: Content returned by `extract_pdf_content`.

    Returns:
        Structured profile dictionary with tentative schema fields.
    """
    log = _get_active_logger()
    log.info("Structuring profile sections")

    sidebar_text = extracted_content.get("sidebar_text", "")
    main_text = extracted_content.get("main_text", "")

    # Clean pagination footers like 'Page 1 of 6'
    cleaned_main = re.sub(r"Page \d+ of \d+\n?", "", main_text).strip()

    sidebar_sections = _parse_sidebar_sections(sidebar_text)

    # Main text split: header before Resumo, and body after Resumo
    if "Resumo\n" in cleaned_main:
        header_part, body_part = cleaned_main.split("Resumo\n", 1)
    elif "Summary\n" in cleaned_main:
        header_part, body_part = cleaned_main.split("Summary\n", 1)
    else:
        header_part, body_part = "", cleaned_main

    header_lines = [line.strip() for line in header_part.splitlines() if line.strip()]
    name = header_lines[0] if header_lines else ""
    headline_with_location_threshold = 2
    if len(header_lines) > headline_with_location_threshold:
        headline = " ".join(header_lines[1:-1])
    elif len(header_lines) == headline_with_location_threshold:
        headline = header_lines[1]
    else:
        headline = ""

    contact = _extract_contact_info(
        sidebar_sections.get("Contato", ""),
        header_lines,
    )

    # Split body into summary, experience, and education
    exp_anchor = "Experiência\n" if "Experiência\n" in body_part else "Experience\n"
    edu_anchor = (
        "Formação acadêmica\n" if "Formação acadêmica\n" in body_part else "Education\n"
    )

    exp_idx = body_part.find(exp_anchor)
    edu_idx = body_part.find(edu_anchor)

    if exp_idx != -1:
        summary = body_part[:exp_idx].strip()
        if edu_idx != -1:
            experience_raw = body_part[exp_idx + len(exp_anchor) : edu_idx].strip()
            education_raw = body_part[edu_idx + len(edu_anchor) :].strip()
        else:
            experience_raw = body_part[exp_idx + len(exp_anchor) :].strip()
            education_raw = ""
    else:
        summary = body_part.strip()
        experience_raw = ""
        education_raw = ""

    skills = _parse_list_lines(sidebar_sections.get("Principais competências", ""))
    languages = _parse_list_lines(sidebar_sections.get("Languages", ""))
    certifications = _parse_list_lines(sidebar_sections.get("Certifications", ""))

    return {
        "name": name,
        "headline": headline,
        "contact": contact,
        "summary": summary,
        "experience": experience_raw,
        "education": education_raw,
        "skills": skills,
        "languages": languages,
        "certifications": certifications,
    }


@task(name="save-extracted-json")
def save_extracted_json(data: dict[str, Any], output_path: str | Path) -> str:
    """Serialize the extracted dataset to a JSON file.

    Args:
        data: The payload dictionary to save.
        output_path: Path where the JSON should be written.

    Returns:
        String path of the generated JSON file.
    """
    log = _get_active_logger()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    json_payload = json.dumps(data, indent=2, ensure_ascii=False)
    destination.write_text(json_payload, encoding="utf-8")
    log.info("Saved extracted JSON data to %s", destination)
    return str(destination)


@flow(name="pdfplumber-cv-extraction-pipeline")
def run_pipeline(
    pdf_path: str,
    output_path: str,
) -> dict[str, Any]:
    """Prefect flow orchestrating PDF extraction and JSON persistence.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path to the destination JSON file.

    Returns:
        The full extracted and structured dictionary.
    """
    log = _get_active_logger()
    log.info("Starting CV extraction flow for %s", pdf_path)

    extracted_content = extract_pdf_content(pdf_path)
    profile = parse_profile_structure(extracted_content)

    result = {
        "source_pdf": str(pdf_path),
        "parser": "pdfplumber",
        "metadata": extracted_content["metadata"],
        "profile": profile,
        "pages": extracted_content["pages"],
        "raw_text": extracted_content["raw_text"],
    }

    save_extracted_json(result, output_path)
    log.info("CV extraction pipeline completed successfully.")
    return result
