"""Prefect flow that extracts a structured profile from a PDF using PyMuPDF."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pymupdf
from prefect import flow, get_run_logger, task

PARSER = "pymupdf"
PARSER_VERSION = list(pymupdf.version)

CONTACT_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"\+?\d[\d\s().-]{8,}\d"),
    "linkedin": re.compile(r"linkedin\.com/in/[^\s]+", re.IGNORECASE),
    "url": re.compile(r"https?://[^\s]+"),
}

SECTION_HEADERS: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "profile", "about me", "about"),
    "experience": ("experience", "professional experience", "work experience", "career history"),
    "education": ("education", "academic background", "academics"),
    "skills": ("skills", "technical skills", "core competencies", "competencies"),
    "languages": ("languages",),
    "certifications": ("certifications", "certificates", "certifications and awards"),
}

_LIST_SECTIONS = frozenset({"experience", "education", "skills", "languages", "certifications"})


@dataclass
class ExtractedPDF:
    """Raw extraction result produced by PyMuPDF."""

    metadata: dict[str, str]
    pages: list[str]
    num_pages: int


@dataclass
class Profile:
    """Tentative structured representation of a CV profile."""

    name: str = ""
    headline: str = ""
    contact: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    experience: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)


@task
def extract_pdf(pdf_path: Path) -> ExtractedPDF:
    """Extract per-page text and document metadata from the PDF."""
    with pymupdf.open(pdf_path) as document:
        pages = [page.get_text() for page in document]
        metadata = document.metadata
    return ExtractedPDF(metadata=metadata, pages=pages, num_pages=len(pages))


@task
def build_profile(pages: list[str]) -> dict[str, object]:
    """Build a tentative structured profile from the extracted page texts."""
    lines = [line.strip() for line in "\n".join(pages).splitlines() if line.strip()]
    profile = Profile()
    body = list(lines)
    if body:
        profile.name = body.pop(0)
    if body:
        profile.headline = body.pop(0)
    profile.contact = _detect_contact(lines)
    for key, section_lines in _split_sections(body):
        if key is None:
            if not profile.summary:
                profile.summary = " ".join(section_lines)
        elif key == "summary":
            profile.summary = " ".join(section_lines)
        elif key in _LIST_SECTIONS:
            setattr(profile, key, section_lines)
    return asdict(profile)


def _detect_contact(lines: list[str]) -> dict[str, str]:
    """Detect contact information from the extracted lines."""
    text = " ".join(lines)
    contact: dict[str, str] = {}
    for kind, pattern in CONTACT_PATTERNS.items():
        if match := pattern.search(text):
            contact[kind] = match.group(0).rstrip(".,;")
    return contact


def _match_section(line: str) -> str | None:
    """Return the normalized section key if the line matches a known header."""
    lowered = line.lower()
    for key, aliases in SECTION_HEADERS.items():
        if lowered in aliases or any(alias in lowered for alias in aliases):
            return key
    return None


def _split_sections(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    """Split flat lines into sections keyed by their detected headers."""
    sections: list[tuple[str | None, list[str]]] = []
    current: str | None = None
    for line in lines:
        key = _match_section(line)
        if key is not None:
            current = key
            sections.append((key, []))
        elif current is None:
            if sections and sections[-1][0] is None:
                sections[-1][1].append(line)
            else:
                sections.append((None, [line]))
        else:
            sections[-1][1].append(line)
    return sections


def assemble_payload(pdf_path: Path, extracted: ExtractedPDF, profile: dict[str, object]) -> dict[str, object]:
    """Combine the extraction artifacts into the final JSON payload."""
    metadata = {"num_pages": extracted.num_pages, **extracted.metadata}
    return {
        "source_pdf": str(pdf_path),
        "parser": PARSER,
        "parser_version": PARSER_VERSION,
        "metadata": metadata,
        "profile": profile,
        "pages": extracted.pages,
    }


@task
def dump_json(payload: dict[str, object], output_path: Path) -> None:
    """Serialize the extraction payload to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@flow
def parse_pdf_flow(pdf_path: Path, output_path: Path = Path("data/profile_extracted_pymupdf.json")) -> dict[str, object]:
    """Run the PDF-to-JSON extraction pipeline end to end."""
    logger = get_run_logger()
    extracted = extract_pdf(pdf_path)
    profile = build_profile(extracted.pages)
    payload = assemble_payload(pdf_path, extracted, profile)
    dump_json(payload, output_path)
    logger.info("Extracted %d pages from %s into %s", extracted.num_pages, pdf_path, output_path)
    return payload


def run() -> None:
    """Run the extraction pipeline from the command line."""
    cli = argparse.ArgumentParser(description="Extract a structured profile from a PDF into JSON using PyMuPDF.")
    cli.add_argument("pdf", nargs="?", default="data/Profile.pdf", help="Path to the input PDF file.")
    cli.add_argument("output", nargs="?", default="data/profile_extracted_pymupdf.json", help="Path to the output JSON file.")
    args = cli.parse_args()
    parse_pdf_flow(Path(args.pdf), Path(args.output))


if __name__ == "__main__":
    run()
