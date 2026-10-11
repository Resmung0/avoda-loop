"""PDF → JSON extraction pipeline using pypdf and Prefect."""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prefect import flow, task
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Constants & module-level compiled regexes
# ---------------------------------------------------------------------------

_MIN_PIPE_COUNT = 2  # minimum "|" chars to qualify as a LinkedIn-style headline

_SECTION_HEADINGS: dict[str, list[str]] = {
    "summary": [
        "summary",
        "about",
        "profile",
        "objective",
        "about me",
        # Portuguese
        "resumo",
        "sobre mim",
        "perfil",
        "objetivo",
    ],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        # Portuguese
        "experiência",
        "experiencia",
        "experiência profissional",
        "experiencia profissional",
        "histórico profissional",
        "historico profissional",
    ],
    "education": [
        "education",
        "academic background",
        "qualifications",
        # Portuguese
        "educação",
        "educacao",
        "formação",
        "formacao",
        "formação acadêmica",
        "formacao academica",
        "escolaridade",
    ],
    "skills": [
        "skills",
        "technical skills",
        "competencies",
        "technologies",
        # Portuguese
        "habilidades",
        "competências",
        "competencias",
        "tecnologias",
        "habilidades técnicas",
        "habilidades tecnicas",
        "principais competências",
        "principais competencias",
    ],
    "languages": [
        "languages",
        "language skills",
        # Portuguese
        "idiomas",
        "línguas",
        "linguas",
    ],
    "certifications": [
        "certifications",
        "certificates",
        "licenses",
        "credentials",
        # Portuguese
        "certificações",
        "certificacoes",
        "certificados",
        "licenças",
        "licencas",
    ],
}

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"[\+]?[\d][\d\s\-\(\)\.]{7,}[\d]")
_URL_RE = re.compile(r"https?://[^\s]+|(?:www\.|linkedin\.com|github\.com)[^\s]*")
_TITLE_RE = re.compile(
    r"engineer|developer|analyst|designer|manager|scientist|architect"
    r"|engenheiro|desenvolvedor|analista|gerente|cientista",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Strip excessive whitespace from *text*."""
    return re.sub(r"\s+", " ", text).strip()


def _detect_section(line: str) -> str | None:
    """Return the canonical section key if *line* looks like a section heading."""
    normalised = line.strip().lower().rstrip(":")
    for key, aliases in _SECTION_HEADINGS.items():
        if normalised in aliases:
            return key
    return None


def _is_contact_line(line: str) -> bool:
    """Return True if *line* contains contact information or a URL."""
    return bool(
        _EMAIL_RE.search(line) or _PHONE_RE.search(line) or _URL_RE.search(line)
    )


def _extract_contact(all_text: str) -> dict[str, Any]:
    """Extract email, phone, and URL contact details from *all_text*.

    Args:
        all_text: Full raw text of the document.

    Returns:
        Dict with keys ``email``, ``phone``, and ``urls``.
    """
    emails = list({m.group() for m in _EMAIL_RE.finditer(all_text)})
    phones = list({_clean(m.group()) for m in _PHONE_RE.finditer(all_text)})
    urls = list({m.group().rstrip(".,)") for m in _URL_RE.finditer(all_text)})
    return {
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
        "urls": urls,
    }


def _extract_name_headline(lines: list[str]) -> tuple[str, str]:
    """Detect the person's name and professional headline from *lines*.

    LinkedIn PDFs typically place a sidebar (contact info, skills, languages)
    before the main content. The name appears immediately before a
    professional headline line that contains multiple ``|`` separators.

    Args:
        lines: Cleaned, non-empty lines from the full document text.

    Returns:
        A ``(name, headline)`` tuple; either may be an empty string if not found.
    """
    for idx, line in enumerate(lines):
        if idx + 1 >= len(lines):
            break
        next_line = lines[idx + 1]
        is_headline = next_line.count("|") >= _MIN_PIPE_COUNT or (
            _TITLE_RE.search(next_line) and "|" in next_line
        )
        is_clean_name = not (_is_contact_line(line) or "|" in line)
        if is_headline and is_clean_name:
            return line, next_line

    # Fallback: first two non-contact lines
    name = ""
    headline = ""
    for line in lines:
        if not _is_contact_line(line):
            if not name:
                name = line
            else:
                headline = line
                break
    return name, headline


def _split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Split *lines* into sections based on heading keywords.

    Args:
        lines: Cleaned document lines.

    Returns:
        Dict mapping section keys to lists of content lines.
    """
    sections: dict[str, list[str]] = {k: [] for k in _SECTION_HEADINGS}
    current_section: str | None = None
    for line in lines:
        key = _detect_section(line)
        if key is not None:
            current_section = key
            continue
        if current_section is not None:
            sections[current_section].append(line)
    return sections


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@task(name="extract-raw-text", description="Read every page of the PDF and return raw text per page.")
def extract_raw_text(pdf_path: Path) -> dict[str, Any]:
    """Extract raw text and PDF metadata from *pdf_path*.

    Args:
        pdf_path: Path to the source PDF file.

    Returns:
        Dict with keys ``metadata``, ``pages``, and ``raw_text``.
    """
    reader = PdfReader(str(pdf_path))
    info = reader.metadata or {}
    metadata: dict[str, Any] = {
        "title": info.get("/Title", ""),
        "author": info.get("/Author", ""),
        "creator": info.get("/Creator", ""),
        "producer": info.get("/Producer", ""),
        "subject": info.get("/Subject", ""),
        "creation_date": str(info.get("/CreationDate", "")),
        "num_pages": len(reader.pages),
    }
    pages: list[str] = [page.extract_text() or "" for page in reader.pages]
    raw_text = "\n".join(pages)
    return {"metadata": metadata, "pages": pages, "raw_text": raw_text}


@task(name="parse-profile", description="Heuristically parse profile fields from raw text.")
def parse_profile(raw_text: str) -> dict[str, Any]:
    """Parse structured profile fields from *raw_text* using heuristics.

    Args:
        raw_text: The full plain-text content of the PDF.

    Returns:
        Dict with keys: name, headline, contact, summary, experience,
        education, skills, languages, certifications.
    """
    lines = [_clean(ln) for ln in raw_text.splitlines() if _clean(ln)]
    contact = _extract_contact(raw_text)
    name, headline = _extract_name_headline(lines)
    sections = _split_sections(lines)

    return {
        "name": name,
        "headline": headline,
        "contact": contact,
        "summary": "\n".join(sections["summary"]),
        "experience": sections["experience"],
        "education": sections["education"],
        "skills": sections["skills"],
        "languages": sections["languages"],
        "certifications": sections["certifications"],
    }


@task(name="build-output", description="Assemble the final output dict.")
def build_output(
    source_pdf: Path,
    metadata: dict[str, Any],
    profile: dict[str, Any],
    pages: list[str],
    raw_text: str,
) -> dict[str, Any]:
    """Assemble all extracted data into the final output structure.

    Args:
        source_pdf: Original PDF path.
        metadata: PDF document metadata.
        profile: Parsed profile fields.
        pages: Per-page raw text list (for audit).
        raw_text: Full concatenated raw text (for audit).

    Returns:
        Final structured dict ready to be serialised to JSON.
    """
    return {
        "source_pdf": str(source_pdf.resolve()),
        "parser": "pypdf",
        "extracted_at": datetime.now(tz=UTC).isoformat(),
        "metadata": metadata,
        "profile": profile,
        "pages": pages,
        "raw_text": raw_text,
    }


@task(name="write-json", description="Write the output dict to a JSON file.")
def write_json(output: dict[str, Any], dest: Path) -> Path:
    """Serialise *output* as pretty-printed JSON and write it to *dest*.

    Args:
        output: The structured extraction result.
        dest: Destination file path.

    Returns:
        The path that was written.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@flow(name="pdf-to-json-pypdf", description="Extract structured JSON from a PDF CV using pypdf.")
def pdf_to_json(
    pdf_path: Path = Path("data/Profile.pdf"),
    output_path: Path = Path("data/profile_extracted_pypdf.json"),
) -> Path:
    """Run the full PDF → JSON extraction pipeline.

    Args:
        pdf_path: Source PDF to parse (default: ``data/Profile.pdf``).
        output_path: Destination JSON file (default: ``data/profile_extracted_pypdf.json``).

    Returns:
        Path to the generated JSON file.
    """
    extracted = extract_raw_text(pdf_path)
    profile = parse_profile(extracted["raw_text"])
    output = build_output(
        source_pdf=pdf_path,
        metadata=extracted["metadata"],
        profile=profile,
        pages=extracted["pages"],
        raw_text=extracted["raw_text"],
    )
    return write_json(output, output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """CLI entry point: parse arguments and run the Prefect flow."""
    parser = argparse.ArgumentParser(description="PDF → JSON extraction pipeline (pypdf).")
    parser.add_argument(
        "--pdf",
        default="data/Profile.pdf",
        help="Path to the source PDF file (default: data/Profile.pdf)",
    )
    parser.add_argument(
        "--output",
        default="data/profile_extracted_pypdf.json",
        help="Destination JSON path (default: data/profile_extracted_pypdf.json)",
    )
    args = parser.parse_args()
    result = pdf_to_json(pdf_path=Path(args.pdf), output_path=Path(args.output))
    print(f"JSON written to: {result}")  # noqa: T201
