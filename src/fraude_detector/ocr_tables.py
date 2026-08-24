"""Deterministic reconstruction of tables already present in GLM-OCR regions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser

from fraude_detector.structured_ocr import StructuredOcrRegion


@dataclass(frozen=True, slots=True)
class ParsedOcrTable:
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _ParsedRow:
    cells: tuple[str, ...]
    contains_header: bool


_COLUMN_ROLE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "value_date",
        (r"\bvalue date\b", r"\bdate (?:de )?valeur\b", r"\bvalutadatum\b"),
    ),
    (
        "transaction_date",
        (
            r"\btransaction date\b",
            r"\bbooking date\b",
            r"\bdate (?:operation|transaction|comptable)\b",
            r"\bbuchungsdatum\b",
            r"\bdatum\b",
            r"^date$",
        ),
    ),
    (
        "description",
        (
            r"\bdescription\b",
            r"\blibelle\b",
            r"\bdesignation\b",
            r"\bparticulars?\b",
            r"\bdetails?\b",
            r"\bmerchant\b",
            r"\bbeneficiaire\b",
            r"\bverwendungszweck\b",
            r"\bbeschreibung\b",
        ),
    ),
    (
        "debit_amount",
        (r"\bdebits?\b", r"\bdebit amount\b", r"\bwithdrawals?\b", r"\bsoll\b"),
    ),
    (
        "credit_amount",
        (r"\bcredits?\b", r"\bcredit amount\b", r"\bdeposits?\b", r"\bhaben\b"),
    ),
    ("balance", (r"\bbalance\b", r"\bsolde\b", r"\bsaldo\b")),
    ("currency", (r"\bcurrency\b", r"\bdevise\b", r"\bwaehrung\b")),
    ("quantity", (r"\bquantity\b", r"\bquantite\b", r"\bqty\b", r"\bqte\b", r"\bmenge\b")),
    (
        "unit_price",
        (
            r"\bunit price\b",
            r"\bprix unitaire\b",
            r"\bprice per unit\b",
            r"\beinzelpreis\b",
        ),
    ),
    (
        "tax_rate",
        (r"\btax rate\b", r"\bvat rate\b", r"\btaux (?:de )?(?:tva|taxe)\b", r"\bmwst ?%"),
    ),
    (
        "tax_amount",
        (r"\btax amount\b", r"\bvat amount\b", r"\bmontant (?:de )?(?:tva|taxe)\b"),
    ),
    ("service_code", (r"\bservice code\b", r"\bcode (?:service|prestation|acte)\b")),
    (
        "product_code",
        (r"\bproduct code\b", r"\bcode produit\b", r"\bproduct reference\b", r"\bsku\b"),
    ),
)


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[_ParsedRow]] = []
        self._table_depth = 0
        self._rows: list[_ParsedRow] = []
        self._cells: list[str] | None = None
        self._cell_fragments: list[str] | None = None
        self._row_contains_header = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._rows = []
        elif self._table_depth == 1 and tag == "tr":
            self._cells = []
            self._row_contains_header = False
        elif self._table_depth == 1 and tag in {"th", "td"} and self._cells is not None:
            self._cell_fragments = []
            self._row_contains_header = self._row_contains_header or tag == "th"
        elif self._cell_fragments is not None and tag == "br":
            self._cell_fragments.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell_fragments is not None:
            self._cell_fragments.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._table_depth == 1 and tag in {"th", "td"} and self._cells is not None:
            self._cells.append(_clean_cell("".join(self._cell_fragments or ())))
            self._cell_fragments = None
        elif self._table_depth == 1 and tag == "tr" and self._cells is not None:
            if any(self._cells):
                self._rows.append(_ParsedRow(tuple(self._cells), self._row_contains_header))
            self._cells = None
            self._cell_fragments = None
        elif tag == "table" and self._table_depth:
            if self._table_depth == 1 and self._rows:
                self.tables.append(self._rows)
            self._table_depth -= 1


def parse_ocr_table(content: str) -> ParsedOcrTable | None:
    """Parse one OCR table without asking the language model to copy its cells."""

    html_table = _parse_html_table(content)
    if html_table is not None:
        return html_table
    return _parse_markdown_table(content)


def table_from_regions(regions: tuple[StructuredOcrRegion, ...]) -> ParsedOcrTable | None:
    """Select the richest table found in the source regions referenced by the model."""

    candidates = tuple(
        table for region in regions if (table := parse_ocr_table(region.content)) is not None
    )
    if not candidates:
        return None
    return max(candidates, key=lambda table: sum(map(len, table.rows)) + len(table.headers))


def infer_table_column_role(header: str, semantic_type: str) -> str:
    """Infer a canonical role only when an OCR header is semantically explicit."""

    normalized = _normalized_header(header)
    if not normalized:
        return "other"
    for role, patterns in _COLUMN_ROLE_PATTERNS:
        if any(re.search(pattern, normalized) for pattern in patterns):
            return role
    if re.search(r"\b(?:line total|total ligne|gesamtpreis)\b", normalized):
        return "line_total"
    if re.search(r"\b(?:amount|montant|betrag|value)\b", normalized):
        return (
            "line_total"
            if semantic_type in {"invoice_lines", "services", "products"}
            else "amount"
        )
    if normalized in {"total", "gesamt"}:
        return (
            "line_total"
            if semantic_type in {"invoice_lines", "services", "products"}
            else "amount"
        )
    return "other"


def _parse_html_table(content: str) -> ParsedOcrTable | None:
    if "<table" not in content.casefold():
        return None
    parser = _TableParser()
    try:
        parser.feed(content)
        parser.close()
    except (ValueError, AssertionError):
        return None
    parsed = tuple(_table_from_rows(rows) for rows in parser.tables)
    tables = tuple(table for table in parsed if table is not None)
    if not tables:
        return None
    return max(tables, key=lambda table: sum(map(len, table.rows)) + len(table.headers))


def _table_from_rows(rows: list[_ParsedRow]) -> ParsedOcrTable | None:
    if not rows:
        return None
    header_index = next((index for index, row in enumerate(rows) if row.contains_header), None)
    headers = rows[header_index].cells if header_index is not None else ()
    body = tuple(row.cells for index, row in enumerate(rows) if index != header_index)
    return ParsedOcrTable(headers=headers, rows=body) if headers or body else None


def _parse_markdown_table(content: str) -> ParsedOcrTable | None:
    lines = tuple(line.strip() for line in content.splitlines() if "|" in line)
    if len(lines) < 2:
        return None
    rows = tuple(_markdown_cells(line) for line in lines)
    separator_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in row)
        ),
        None,
    )
    if separator_index != 1 or not rows[0]:
        return None
    body = tuple(row for row in rows[2:] if any(row))
    return ParsedOcrTable(headers=rows[0], rows=body)


def _markdown_cells(line: str) -> tuple[str, ...]:
    stripped = line.strip().strip("|")
    return tuple(_clean_cell(cell) for cell in stripped.split("|"))


def _clean_cell(value: str) -> str:
    return " ".join(value.split())[:500]


def _normalized_header(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9%]+", " ", ascii_value.casefold()).strip()
