"""
docx_helpers.py - small helpers that give a python-docx document the look of an IEEE paper:
Times New Roman 10 pt, two columns, numbered figures ("Fig. 1."), numbered tables ("TABLE I"),
numbered equations, and an IEEE reference list.

Text markup understood by add_text():  **bold**   *italic*   `code`   _{subscript}   ^{superscript}
"""
import copy
import re

from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"
CODE_FONT = "Courier New"
COLUMN_WIDTH = 3.375      # inches, one column of the two-column layout
FULL_WIDTH = 7.0          # inches, text width
TOKEN = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`|_\{.+?\}|\^\{.+?\})")
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


# -----------------------------------------------------------------------------
# styles
# -----------------------------------------------------------------------------
def set_font(style_or_run, size=None, bold=None, italic=None, name=FONT):
    font = style_or_run.font
    font.name = name
    element = style_or_run.element.rPr if hasattr(style_or_run.element, "rPr") else None
    if element is not None:
        rfonts = element.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            element.insert(0, rfonts)
        for attribute in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(attribute), name)
    if size:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if italic is not None:
        font.italic = italic


def make_style(doc, name, size=10, bold=False, italic=False, align=None, before=0, after=0,
               first_indent=None, left_indent=None, keep_next=False, line=1.0, outline=None):
    style = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style.base_style = doc.styles["Normal"]
    set_font(style, size, bold, italic)
    paragraph = style.paragraph_format
    paragraph.space_before = Pt(before)
    paragraph.space_after = Pt(after)
    paragraph.line_spacing = line
    paragraph.keep_with_next = keep_next
    paragraph.widow_control = True
    if align is not None:
        paragraph.alignment = align
    if first_indent is not None:
        paragraph.first_line_indent = Inches(first_indent)
    if left_indent is not None:
        paragraph.left_indent = Inches(left_indent)
    if outline is not None:
        properties = style.element.get_or_add_pPr()
        level = OxmlElement("w:outlineLvl")
        level.set(qn("w:val"), str(outline))
        properties.append(level)
    return style


def define_styles(doc):
    J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
    make_style(doc, "R Body", 10, align=J, after=3, first_indent=0.17)
    make_style(doc, "R Body Flush", 10, align=J, after=3)
    make_style(doc, "R Chapter", 11, bold=True, align=C, before=10, after=5, keep_next=True, outline=0)
    make_style(doc, "R Heading 2", 10, italic=True, align=L, before=6, after=2, keep_next=True, outline=1)
    make_style(doc, "R Abstract", 9, bold=True, align=J, after=3)
    make_style(doc, "R Caption", 8, align=C, before=2, after=6)
    make_style(doc, "R Table Caption", 8, align=C, before=4, after=3, keep_next=True)
    make_style(doc, "R Figure", 10, align=C, before=2, after=0, keep_next=True)
    make_style(doc, "R Table Cell", 8, align=L)
    make_style(doc, "R Equation", 10, align=L, before=2, after=3)
    make_style(doc, "R Bullet", 10, align=J, after=1, left_indent=0.22, first_indent=-0.15)
    make_style(doc, "R Ref", 8, align=J, after=1.5, left_indent=0.3, first_indent=-0.3)
    make_style(doc, "R Keywords", 9, italic=True, bold=True, align=J, after=4)
    make_style(doc, "R Spacer", 1, line=1.0)            # carries a section break; takes almost no height


# -----------------------------------------------------------------------------
# text
# -----------------------------------------------------------------------------
def add_runs(paragraph, text, size=None):
    """Append text to a paragraph, honouring the small markup described at the top of the file."""
    for piece in TOKEN.split(text):
        if not piece:
            continue
        run = None
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            run = paragraph.add_run(piece[2:-2])
            run.bold = True
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            run = paragraph.add_run(piece[1:-1])
            run.italic = True
        elif piece.startswith("`") and piece.endswith("`"):
            run = paragraph.add_run(piece[1:-1])
            set_font(run, (size or 10) - 1, name=CODE_FONT)
            continue
        elif piece.startswith("_{"):
            run = paragraph.add_run(piece[2:-1])
            run.font.subscript = True
        elif piece.startswith("^{"):
            run = paragraph.add_run(piece[2:-1])
            run.font.superscript = True
        else:
            run = paragraph.add_run(piece)
        set_font(run, size)
    return paragraph


def add_text(doc, text, style="R Body"):
    paragraph = doc.add_paragraph(style=style)
    add_runs(paragraph, text)
    return paragraph


def add_bullets(doc, items):
    for item in items:
        paragraph = doc.add_paragraph(style="R Bullet")
        add_runs(paragraph, "• " + item)


def add_chapter(doc, title):
    return doc.add_paragraph(title, style="R Chapter")


def add_heading2(doc, title):
    return doc.add_paragraph(title, style="R Heading 2")


class Counter:
    figures = 0
    tables = 0
    equations = 0


def add_equation(doc, text, width=COLUMN_WIDTH):
    """Centered equation with its number at the right margin: (n)."""
    Counter.equations += 1
    paragraph = doc.add_paragraph(style="R Equation")
    tabs = paragraph.paragraph_format.tab_stops
    tabs.add_tab_stop(Inches(width / 2), WD_TAB_ALIGNMENT.CENTER)
    tabs.add_tab_stop(Inches(width), WD_TAB_ALIGNMENT.RIGHT)
    paragraph.add_run("\t")
    add_runs(paragraph, text)
    paragraph.add_run(f"\t({Counter.equations})")
    return paragraph


# -----------------------------------------------------------------------------
# figures and tables
# -----------------------------------------------------------------------------
def add_figure(doc, path, caption, width=COLUMN_WIDTH - 0.05):
    Counter.figures += 1
    holder = doc.add_paragraph(style="R Figure")
    holder.add_run().add_picture(str(path), width=Inches(width))
    label = doc.add_paragraph(style="R Caption")
    bold_part = label.add_run(f"Fig. {Counter.figures}. ")
    set_font(bold_part, 8, bold=True)
    add_runs(label, caption, size=8)
    return Counter.figures


def set_cell_border(cell, **edges):
    """edges: top/bottom/left/right = size in eighths of a point (0 or None = none)."""
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right"):
        size = edges.get(edge)
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single" if size else "nil")
        if size:
            element.set(qn("w:sz"), str(size))
            element.set(qn("w:color"), "000000")
        element.set(qn("w:space"), "0")
        borders.append(element)


def add_table(doc, header, rows, widths, caption, align=None, bold_rows=(), size=8):
    """IEEE-style table: caption above, rules above/below the header and below the last row."""
    Counter.tables += 1
    title = doc.add_paragraph(style="R Table Caption")
    first = title.add_run(f"TABLE {ROMAN[Counter.tables - 1]}")
    set_font(first, 8)
    first.font.small_caps = True
    title.add_run().add_break()
    add_runs(title, caption, size=8)

    table = doc.add_table(rows=1 + len(rows), cols=len(header))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    # python-docx sizes the grid from the page width; give it the real column widths, fixed layout
    for index, width in enumerate(widths):
        table.columns[index].width = Inches(width)
    properties = table._tbl.tblPr          # (table.autofit = False above already wrote the fixed layout)
    total = properties.find(qn("w:tblW"))
    if total is None:
        total = OxmlElement("w:tblW")
        properties.append(total)
    total.set(qn("w:w"), str(int(sum(widths) * 1440)))
    total.set(qn("w:type"), "dxa")
    align = align or ["L"] + ["C"] * (len(header) - 1)
    for r, values in enumerate([header] + list(rows)):
        for c, value in enumerate(values):
            cell = table.cell(r, c)
            cell.width = Inches(widths[c])
            paragraph = cell.paragraphs[0]
            paragraph.style = doc.styles["R Table Cell"]
            paragraph.alignment = {"L": WD_ALIGN_PARAGRAPH.LEFT, "C": WD_ALIGN_PARAGRAPH.CENTER,
                                   "R": WD_ALIGN_PARAGRAPH.RIGHT}[align[c]]
            paragraph.paragraph_format.keep_with_next = r < len(rows)       # keep the table in one piece
            add_runs(paragraph, f"**{value}**" if (r == 0 or (r - 1) in bold_rows) and value else str(value), size=size)
            set_cell_border(cell, top=8 if r == 0 else None, bottom=6 if r in (0, len(rows)) else None,
                            left=None, right=None)
            margins = cell._tc.get_or_add_tcPr()
            cell_margins = OxmlElement("w:tcMar")
            for side, value_dxa in (("top", 15), ("left", 40), ("bottom", 15), ("right", 40)):
                node = OxmlElement(f"w:{side}")
                node.set(qn("w:w"), str(value_dxa))
                node.set(qn("w:type"), "dxa")
                cell_margins.append(node)
            margins.append(cell_margins)
    spacer = doc.add_paragraph(style="R Caption")
    spacer.paragraph_format.space_after = Pt(2)
    spacer.paragraph_format.space_before = Pt(0)
    return table


# -----------------------------------------------------------------------------
# sections: two columns normally, one column for wide figures
# -----------------------------------------------------------------------------
def set_columns(section, count):
    sect_pr = section._sectPr
    for old in sect_pr.findall(qn("w:cols")):
        sect_pr.remove(old)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), "360")
    number = sect_pr.find(qn("w:pgNumType"))
    if number is not None:
        number.addnext(cols)
    else:
        sect_pr.append(cols)
    if number is not None and number.get(qn("w:start")) is not None and not getattr(section, "_keep_start", False):
        pass


FOOTER_REFERENCE = []          # the first body section's footer reference, reused by every later section


def new_section(doc, columns, kind=WD_SECTION.CONTINUOUS, restart_numbers=False):
    # Two tiny paragraphs end the old section. Without them LibreOffice does not balance the two
    # columns before a continuous break and pushes the next section onto a new page.
    doc.add_paragraph(style="R Spacer")
    section = doc.add_section(kind)
    doc.paragraphs[-1].style = doc.styles["R Spacer"]
    if FOOTER_REFERENCE:           # name the same footer explicitly, so every renderer keeps one page style
        section._sectPr.insert(0, copy.deepcopy(FOOTER_REFERENCE[0]))
    number = section._sectPr.find(qn("w:pgNumType"))
    if number is not None:
        if restart_numbers:
            number.set(qn("w:start"), "1")
        elif number.get(qn("w:start")) is not None:
            del number.attrib[qn("w:start")]
    set_columns(section, columns)
    return section


class Wide:
    """with Wide(doc): ...   puts the content in a full-width (one column) strip, then returns to two columns."""
    def __init__(self, doc):
        self.doc = doc

    def __enter__(self):
        new_section(self.doc, 1)

    def __exit__(self, *exc):
        new_section(self.doc, 2)


def add_page_number_footer(section, size=9):
    section.footer.is_linked_to_previous = False
    FOOTER_REFERENCE.clear()
    FOOTER_REFERENCE.append(copy.deepcopy(section._sectPr.find(qn("w:footerReference"))))
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        run = paragraph.add_run()
        set_font(run, size)
        if kind:
            element = OxmlElement("w:fldChar")
            element.set(qn("w:fldCharType"), kind)
        else:
            element = OxmlElement("w:instrText")
            element.set(qn("xml:space"), "preserve")
            element.text = text
        run._r.append(element)
