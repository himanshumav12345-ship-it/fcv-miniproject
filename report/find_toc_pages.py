"""Read the rendered PDF and write the body page number of each heading to report/toc_pages.json."""
import json
import subprocess
from pathlib import Path

PDF = Path(__file__).parent / "FCV_Mini_Project_Report.pdf"
FRONT_PAGES = 3
count = int([l for l in subprocess.run(["pdfinfo", str(PDF)], capture_output=True, text=True).stdout.splitlines()
             if l.startswith("Pages")][0].split()[1])
targets = {"ABSTRACT": "ABSTRACT", "1": "CHAPTER 1: INTRODUCTION", "2": "CHAPTER 2: PROBLEM STATEMENT",
           "3": "CHAPTER 3: METHODOLOGY", "4": "CHAPTER 4: RESULTS", "5": "CHAPTER 5: Lab-1",
           "6": "CHAPTER 6: CONCLUSION", "7": "CHAPTER 7: REFERENCES"}
pages = {}
for physical in range(FRONT_PAGES + 1, count + 1):
    text = subprocess.run(["pdftotext", "-f", str(physical), "-l", str(physical), "-layout", str(PDF), "-"],
                          capture_output=True, text=True).stdout
    for key, needle in targets.items():
        if key not in pages and needle in text:
            pages[key] = physical - FRONT_PAGES
print(pages)
assert len(pages) == len(targets), "a heading was not found"
(Path(__file__).parent / "toc_pages.json").write_text(json.dumps(pages))
