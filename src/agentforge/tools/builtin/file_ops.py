"""Built-in file reader and writer tools."""

from pathlib import Path

from langchain_core.tools import tool


@tool
def file_reader(file_path: str) -> str:
    """Read the contents of a local file.

    Use this tool to read text files, reports, CSVs, or other documents.
    Input should be a file path (relative or absolute).
    Returns the file contents as text (up to 10,000 characters).
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(path))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                return text[:10000]
            except Exception as e:
                return f"PDF read error: {e}"
        return path.read_text(encoding="utf-8", errors="replace")[:10000]
    except Exception as e:
        return f"File read error: {e}"


@tool
def file_writer(file_path: str, content: str) -> str:
    """Write content to a local file.

    Use this tool to save reports, outputs, or any text content to a file.
    Input: file_path (string) and content (string to write).
    Returns a confirmation message.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"File write error: {e}"


FILE_READER_TOOL_TYPES = [
    "file reader",
    "document reader",
    "PDF reader",
    "text reader",
    "report reader",
    "file loader",
]

FILE_WRITER_TOOL_TYPES = [
    "file writer",
    "report generator",
    "document creator",
    "output writer",
    "file saver",
]
