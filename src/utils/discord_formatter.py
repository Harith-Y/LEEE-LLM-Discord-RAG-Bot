"""
Discord-friendly text formatter.

Converts Markdown tables, HTML tags, and other unsupported formatting
into text that renders cleanly in Discord messages.
"""
import re
import logging

logger = logging.getLogger(__name__)


def format_for_discord(text: str) -> str:
    """
    Post-process LLM output so it displays correctly in Discord.

    Handles:
    - <br> / <br/> → newline
    - Markdown tables → indented, labelled rows
    - Remaining HTML tags → stripped
    - Excessive blank lines → collapsed

    Args:
        text: Raw LLM response text

    Returns:
        Discord-friendly text
    """
    if not text:
        return text

    # 1. Replace <br> variants with newlines (before table processing)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # 2. Convert Markdown tables to Discord-friendly format
    text = _convert_tables(text)

    # 3. Strip any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # 4. Strip source citations the LLM may have included
    text = _strip_source_citations(text)

    # 5. Collapse runs of 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _strip_source_citations(text: str) -> str:
    """Remove source/file citations that the LLM may include from retrieved chunks."""
    # Remove patterns like ([Source 1](filename.md)), ([Source 12](file.md))
    text = re.sub(r'\s*\(\[Source\s*\d+\]\([^)]*\)\)', '', text)
    # Remove patterns like [Source 1](filename.md) standalone
    text = re.sub(r'\[Source\s*\d+\]\([^)]*\)', '', text)
    # Remove patterns like (Source: filename.ext) or (Source 1: filename.ext)
    text = re.sub(r'\s*\(Source\s*\d*:\s*[^)]*\)', '', text)
    # Remove patterns like [Source 1: filename.ext]
    text = re.sub(r'\[Source\s*\d+:\s*[^]]*\]', '', text)
    return text


def _convert_tables(text: str) -> str:
    """
    Find Markdown tables in *text* and replace each one with a
    Discord-friendly representation using bold headers as labels.

    A Markdown table looks like:
        | H1 | H2 | H3 |
        |----|----|----|
        | a  | b  | c  |
        | d  | e  | f  |
    """
    lines = text.split('\n')
    result_lines: list[str] = []
    i = 0

    while i < len(lines):
        # Detect a table header row: starts and ends with |
        if _is_table_row(lines[i]):
            table_lines = _collect_table(lines, i)
            if len(table_lines) >= 2:  # At least header + separator
                converted = _table_to_discord(table_lines)
                result_lines.append(converted)
                i += len(table_lines)
                continue
        result_lines.append(lines[i])
        i += 1

    return '\n'.join(result_lines)


def _is_table_row(line: str) -> bool:
    """Check if a line looks like a Markdown table row."""
    stripped = line.strip()
    return stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 2


def _is_separator_row(line: str) -> bool:
    """Check if a line is a Markdown table separator (|---|---|)."""
    stripped = line.strip()
    # Remove pipes, spaces, dashes, colons — if nothing is left, it's a separator
    cleaned = re.sub(r'[|\s\-:]', '', stripped)
    return len(cleaned) == 0 and '|' in stripped


def _collect_table(lines: list[str], start: int) -> list[str]:
    """Collect consecutive table rows starting from *start*."""
    table = []
    for j in range(start, len(lines)):
        if _is_table_row(lines[j]) or _is_separator_row(lines[j]):
            table.append(lines[j])
        else:
            break
    return table


def _parse_row(line: str) -> list[str]:
    """Split a table row into cell values (stripped)."""
    # Remove leading/trailing pipe then split
    stripped = line.strip()
    if stripped.startswith('|'):
        stripped = stripped[1:]
    if stripped.endswith('|'):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split('|')]


def _table_to_discord(table_lines: list[str]) -> str:
    """
    Convert collected Markdown table lines into a Discord-friendly format.

    Each data row becomes a block:
        **Header1:** value1
        **Header2:** value2
        ...
    Rows are separated by a blank line and a thin divider.
    """
    # Extract headers from the first row
    headers = _parse_row(table_lines[0])

    # Find data rows (skip separator rows)
    data_rows = []
    for line in table_lines[1:]:
        if not _is_separator_row(line):
            data_rows.append(_parse_row(line))

    if not data_rows:
        # Unusual: table with headers but no data — just return headers as bold list
        return '\n'.join(f'**{h}**' for h in headers if h)

    blocks: list[str] = []
    for row in data_rows:
        entries: list[str] = []
        for col_idx, cell in enumerate(row):
            if not cell:
                continue
            header = headers[col_idx] if col_idx < len(headers) else f"Column {col_idx + 1}"
            entries.append(f'**{header}:** {cell}')
        if entries:
            blocks.append('\n'.join(entries))

    return '\n\n'.join(blocks)
