"""Ingest govinfo bulk XML for FAR/DFARS into versioned chunks."""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

GOVINFO_BASE = "https://www.govinfo.gov/bulkdata/CFR"
TITLE = 48
YEARS = range(2017, 2026)  # 2017-2025 inclusive
VOLUMES = range(1, 8)  # FAR is parts 1-53 in vols 1-2, DFARS in 6-7

NS = {"cfr": "http://www.gpo.gov/fdsys/pkg/CFR-2023-title48-vol1/xml/CFR-2023-title48-vol1.xml"}

SECTION_RE = re.compile(r"(\d+(?:\.\d+)*(?:-[A-Za-z0-9]+)?)")

def download_year(year: int, dest_dir: Path) -> list[Path]:
    """Download all volumes for a year."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for vol in VOLUMES:
        url = f"{GOVINFO_BASE}/{year}/title-48/CFR-{year}-title48-vol{vol}.xml"
        dest = Path(dest_dir) / f"CFR-{year}-title48-vol{vol}.xml"
        if not dest.exists():
            print(f"Downloading {year} vol {vol}...")
            urllib.request.urlretrieve(url, dest)
        files.append(dest)
    return files

def parse_xml_to_sections(xml_path: Path, year: int) -> list[dict[str, Any]]:
    """Extract sections with text from a CFR volume XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sections = []
    for elem in root.iter():
        if elem.tag.endswith("SECTION") or elem.tag.endswith("SECTNO"):
            sectno = elem.findtext("SECTNO") or elem.findtext("SECTNO")
            subject = elem.findtext("SUBJECT")
            text_parts = []
            for child in elem:
                if child.tag.endswith("P") or child.tag.endswith("FP"):
                    text = "".join(child.itertext()).strip()
                    if text:
                        text_parts.append(text)
            if text_parts:
                sections.append({
                    "section": sectno,
                    "subject": subject or "",
                    "text": "\n".join(text_parts),
                    "year": year,
                })
    return sections

def main(year_start: int = 2017, year_end: int = 2025, out_dir: Path = Path("data/raw")) -> None:
    """Download and extract all sections for given year range."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for year in range(year_start, year_end + 1):
        files = download_year(year, Path("data/raw"))
        all_sections = []
        for f in files:
            sections = parse_xml_to_sections(f, year)
            for s in sections:
                s["year"] = year
            all_sections.extend(sections)
        
        # Save sections for this year
        out_file = Path("data") / f"sections_{year}.json"
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(all_sections, fh, ensure_ascii=False, indent=2)
        print(f"Year {year}: extracted {len(all_sections)} sections -> {out_file}")
    
    print("Ingest complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        main(int(sys.argv[1]), int(sys.argv[2]))
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]), int(sys.argv[1]))
    else:
        main(2017, 2025)
