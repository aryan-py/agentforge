"""Built-in database query tool — SQLite SELECT-only."""

import sqlite3

from langchain_core.tools import tool


@tool
def database_query(query: str, db_path: str = "data/agentforge.db") -> str:
    """Query a SQLite database using SQL SELECT statements.

    Use this tool to retrieve structured data from databases or CSV-like tables.
    Input: a SQL SELECT query string (only SELECT is allowed for safety).
    Returns query results as formatted text.
    """
    query_stripped = query.strip().upper()
    if not query_stripped.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for safety."

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query returned no results."

        columns = list(rows[0].keys())
        header = " | ".join(columns)
        separator = "-" * len(header)
        result_rows = [" | ".join(str(row[col]) for col in columns) for row in rows[:50]]
        return f"{header}\n{separator}\n" + "\n".join(result_rows)
    except Exception as e:
        return f"Database query error: {e}"


TOOL_TYPES = [
    "database query",
    "SQL",
    "data retrieval",
    "structured data",
    "spreadsheet",
    "table lookup",
    "relational data",
    "data access",
]
