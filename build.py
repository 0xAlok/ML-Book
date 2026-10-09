#!/usr/bin/env python3
"""Build the reading outputs for the deep learning prep book.

One command:  python3 build.py

Reads chapters/*.md + solutions/*.md (never modified) and generates:
  build/web/  -> Quarto book -> build/web/_book/  (d2l-style HTML: nav, search,
                 MathJax, per-problem collapsible solutions)
  build/pdf/  -> Quarto book -> build/pdf/_book/  (print PDF, solutions inline
                 after each chapter)

Everything under build/ is generated; only this script is committed.
Requires: quarto >= 1.4 and a LaTeX engine for the PDF (quarto install tinytex).
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

BOOK_DIR = Path(__file__).resolve().parent
CHAPTERS_DIR = BOOK_DIR / "chapters"
SOLUTIONS_DIR = BOOK_DIR / "solutions"
BUILD_DIR = BOOK_DIR / "build"

PARTS = [
    ("Part I \u2014 Mathematical Foundations", range(1, 14)),
    ("Part II \u2014 Probability and Statistics", range(14, 22)),
    ("Part III \u2014 Machine Learning Foundations", range(22, 26)),
    ("Part IV \u2014 Machine Learning Techniques", range(26, 36)),
    ("Part V \u2014 Machine Learning in Practice", range(36, 41)),
    ("Part VI \u2014 Bridge to Deep Learning", range(41, 48)),
]

BOOK_TITLE = "Foundations to Deep Learning"
BOOK_SUBTITLE = "From foundations to deep learning \u2014 a one-stop study book"


def chapter_files():
    files = sorted(CHAPTERS_DIR.glob("[0-9][0-9]-*.md"))
    if len(files) != 47:
        sys.exit(f"expected 47 chapters, found {len(files)}")
    return files


def sanitize(text):
    # Quarto scans for --- YAML blocks anywhere in a document, so a markdown
    # horizontal rule '---' gets misparsed as YAML (and '**2.**' inside it as a
    # YAML alias -> build error). '***' renders identically and is safe.
    return re.sub(r"(?m)^---\s*$", "***", text)


def fix_asset_paths(text, prefix):
    # Chapter markdown references assets/ relative to chapters/. The generated
    # files live deeper: build/pdf/*.qmd needs '../chapters/assets',
    # build/web/_book/*.html needs '../../chapters/assets'.
    return sanitize(text).replace("](assets/", f"]({prefix}/")


def demote_headers(text):
    # '# ' -> '## ', '## ' -> '### ', ... (solutions get appended after the
    # chapter body in the PDF, so they sit one level below the chapter title)
    def _demote(m):
        return "#" + m.group(1)
    return re.sub(r"^(#{1,5})(?=\s)", _demote, text, flags=re.M)


def split_problems(solutions_text, name):
    """Split a solutions file into [(problem_no, title, body)].

    Handles every marker format found in solutions/:
      A:  '## Problem N'
      A2: '## Solution N' (optionally '## Solution N \u2014 title')
      B:  'N. ...' numbered paragraphs
      C:  '**N.** ...' / '**N. (title)** ...' bold-numbered paragraphs
    """
    problems = []

    def _check(problems):
        nums = [int(n) for n, _, _ in problems]
        if nums != list(range(1, len(nums) + 1)):
            sys.exit(f"{name}: problem numbers not sequential 1..N: {nums}")
        return problems

    # Formats A / A2 / D: '## Problem N', '## Solution N [title]', '## N. title'
    parts = re.split(r"^##\s+(?:(?:Problem|Solution)\s+)?(\d+)"
                     r"\s*[.\u2014:\-]?\s*(.*?)\s*$",
                     solutions_text, flags=re.M)
    if len(parts) > 1:
        # parts[0] is the '# Solutions ...' header chunk; drop it.
        for i in range(1, len(parts), 3):
            problems.append((parts[i], parts[i + 1].strip(),
                             parts[i + 2].strip()))
        return _check(problems)

    # Formats B / C: 'N. ' (whitespace after the dot, so '0.3' inside math
    # environments never matches) or '**N.**' / '**N. (title)**' at line start
    marker = r"^(?=(?:\*\*\d+\.)|(?:\d+\.\s))"
    parts = re.split(marker, solutions_text, flags=re.M)
    for p in parts[1:]:
        m = re.match(r"^(?:\*\*)?(\d+)\.(?:\*\*)?(.*)$", p, flags=re.S)
        if not m:
            continue
        first_line, rest = m.group(2).split("\n", 1) \
            if "\n" in m.group(2) else (m.group(2), "")
        title = first_line.strip(" *()")
        problems.append((m.group(1), title, p.strip()))
    if not problems:
        sys.exit(f"{name}: no problems parsed")
    return _check(problems)
    parts = re.split(r"^## Problem (\d+)\s*$", solutions_text, flags=re.M)
    # parts[0] is the '# Solutions \u2014 Chapter N' header chunk; drop it.
    problems = []
    for i in range(1, len(parts), 2):
        problems.append((parts[i], parts[i + 1].strip()))
    return problems


def build_web_qmd(chapter_text, problems):
    out = [fix_asset_paths(chapter_text, "../../chapters/assets").rstrip(), "",
           "## Solutions", "",
           "Attempt each problem first, then click a problem to reveal its "
           "worked solution.", ""]
    for num, title, body in problems:
        out.append('::: {.callout-tip collapse="true"}')
        out.append(f"## Solution {num}" + (f" \u2014 {title}" if title else ""))
        out.append("")
        out.append(fix_asset_paths(body, "../../chapters/assets").rstrip())
        out.append(':::')
        out.append("")
    return localize_remote_images("\n".join(out), "svg", "../../chapters/assets")


def build_pdf_qmd(chapter_text, solutions_text):
    text = (fix_asset_paths(chapter_text, "../chapters/assets").rstrip() + "\n\n"
            + demote_headers(fix_asset_paths(solutions_text,
                                             "../chapters/assets")).strip()
            + "\n")
    return localize_remote_images(text, "pdf", "../chapters/assets")


# Remote images referenced by chapters. Vendored at build time into
# build/chapters/assets/ so the outputs don't depend on the network;
# SVGs are converted to PDF for the LaTeX build.
REMOTE_IMAGES = {
    "https://d2l.ai/_images/vec-add.svg": "vec-add.svg",  # ch 2, CC-BY-SA d2l.ai
}


def vendor_remote_images(assets_dir):
    import urllib.request
    for url, fname in REMOTE_IMAGES.items():
        dest = assets_dir / fname
        if not dest.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
        if fname.endswith(".svg"):
            pdf = dest.with_suffix(".pdf")
            if not pdf.exists():
                try:
                    import cairosvg
                except ImportError:
                    sys.exit("cairosvg is required to build the PDF "
                             "(pip install cairosvg)")
                print(f"converting {fname} -> {pdf.name}")
                cairosvg.svg2pdf(url=str(dest), write_to=str(pdf))


def localize_remote_images(text, fmt, prefix):
    for url, fname in REMOTE_IMAGES.items():
        local = fname if not fname.endswith(".svg") else (
            fname if fmt == "svg" else fname[:-4] + ".pdf")
        text = text.replace(url, f"{prefix}/{local}")
    return text


INDEX_QMD = f"""# {BOOK_TITLE}

*{BOOK_SUBTITLE}*

This book takes you from mathematical foundations (linear algebra, calculus,
probability) through classical machine learning to neural networks and
transformers \u2014 everything needed before the core deep learning and AI courses.

## How to study from this book

i) Read a chapter front to back: concept first, then the worked `eg` blocks.
ii) Hard topics carry a **"Basically, ..."** line \u2014 the same idea restated
as simply as possible.
iii) Do the **Practice set** at the end of each chapter *before* looking at
solutions: on the web, each solution is hidden behind a click; in the PDF,
solutions follow each chapter.

There is nowhere else you need to go. Links out appear only where the book
itself could not explain something sufficiently.
"""


def write_quarto_yml(path, fmt, qmd_files):
    lines = [
        "project:",
        "  type: book",
        "  output-dir: _book",
        "",
        "book:",
        f'  title: "{BOOK_TITLE}"',
        f'  subtitle: "{BOOK_SUBTITLE}"',
        "  chapters:",
        "    - index.qmd",
    ]
    by_num = {}
    for f in qmd_files:
        m = re.match(r"(\d+)-", f.stem)
        if m:
            by_num[int(m.group(1))] = f
    for part_title, nums in PARTS:
        lines.append(f'    - part: "{part_title}"')
        lines.append("      chapters:")
        for n in nums:
            f = by_num.get(n)
            if f is None:
                sys.exit(f"chapter {n:02d} missing from generated qmd files")
            lines.append(f"        - {f.name}")
    lines += ["", "format:"]
    if fmt == "html":
        lines += [
            "  html:",
            "    toc: true",
            "    toc-depth: 3",
            "    number-sections: false",
            "    search: true",
            "    html-math-method: mathjax",
            "    code-copy: true",
        ]
    else:
        lines += [
            "  pdf:",
            "    documentclass: book",
            "    toc: true",
            "    toc-depth: 3",
            "    number-sections: false",
            "    geometry: margin=1in",
            "    include-in-header:",
            "      text: |",
            "        \\usepackage{fvextra}",
            "        \\DefineVerbatimEnvironment{Highlighting}{Verbatim}",
            "          {breaklines,breakanywhere,commandchars=\\\\\\{\\}}",
        ]
    path.write_text("\n".join(lines) + "\n")


def generate():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    web_dir = BUILD_DIR / "web"
    pdf_dir = BUILD_DIR / "pdf"
    web_dir.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)

    # Generated files reference the mirrored assets relatively:
    #   build/pdf/*.qmd          -> ../chapters/assets/
    #   build/web/_book/*.html   -> ../../chapters/assets/
    shutil.copytree(CHAPTERS_DIR / "assets", BUILD_DIR / "chapters" / "assets")
    vendor_remote_images(BUILD_DIR / "chapters" / "assets")

    web_qmds, pdf_qmds = [], []
    for ch in chapter_files():
        slug = ch.stem
        sol = SOLUTIONS_DIR / f"{slug}.md"
        if not sol.exists():
            sys.exit(f"missing solutions file for {slug}")
        chapter_text = ch.read_text()
        problems = split_problems(sol.read_text(), sol.name)

        web_qmd = web_dir / f"{slug}.qmd"
        web_qmd.write_text(build_web_qmd(chapter_text, problems))
        web_qmds.append(web_qmd)

        pdf_qmd = pdf_dir / f"{slug}.qmd"
        pdf_qmd.write_text(build_pdf_qmd(chapter_text, sol.read_text()))
        pdf_qmds.append(pdf_qmd)

    (web_dir / "index.qmd").write_text(INDEX_QMD)
    (pdf_dir / "index.qmd").write_text(INDEX_QMD)
    write_quarto_yml(web_dir / "_quarto.yml", "html",
                     web_qmds + [web_dir / "index.qmd"])
    write_quarto_yml(pdf_dir / "_quarto.yml", "pdf",
                     pdf_qmds + [pdf_dir / "index.qmd"])
    print(f"generated {len(web_qmds)} web + {len(pdf_qmds)} pdf chapter files")


def render(directory, label):
    print(f"--- rendering {label} ---")
    r = subprocess.run(["quarto", "render"], cwd=directory,
                       capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        sys.exit(f"quarto render failed for {label}")
    print(f"--- {label} done ---")


def main():
    html_only = "--html-only" in sys.argv
    pdf_only = "--pdf-only" in sys.argv
    generate()
    if not pdf_only:
        render(BUILD_DIR / "web", "HTML book")
    if not html_only:
        render(BUILD_DIR / "pdf", "PDF book")
        pdfs = list((BUILD_DIR / "pdf" / "_book").glob("*.pdf"))
        if not pdfs:
            sys.exit("no PDF produced")
        dest = BOOK_DIR.parent / "files" / "deep-learning-prep-book.pdf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdfs[0], dest)
        print(f"PDF copied to {dest}")


if __name__ == "__main__":
    main()
