"""Basic tool implementations for Ada's autonomous capabilities."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict


class WebSearchTool:
    def __init__(self) -> None:
        self.result_cache: Dict[str, str] = {}

    def execute(self, query: str, max_results: int = 5) -> str:
        if query in self.result_cache:
            return self.result_cache[query]

        try:
            result = self._search_with_system_tools(query, max_results)
            self.result_cache[query] = result
            return result
        except Exception as e:
            return f"Search failed: {str(e)}. Please try a web search manually."

    def _search_with_system_tools(self, query: str, max_results: int) -> str:
        if self._is_macos():
            return self._mdfind_search(query)
        elif self._has_google_cli():
            return self._google_search(query, max_results)
        else:
            return self._simulate_search(query)

    def _is_macos(self) -> bool:
        try:
            subprocess.run(["command", "-v", "mdfind"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _has_google_cli(self) -> bool:
        try:
            subprocess.run(["command", "-v", "google"], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _mdfind_search(self, query: str) -> str:
        try:
            result = subprocess.run(
                ["mdfind", query], capture_output=True, text=True, timeout=30
            )
            files = result.stdout.strip().split("\n")[:10]
            if files and files[0]:
                return f"Local files found for '{query}':\n" + "\n".join(files[:5])
            else:
                return self._simulate_search(query)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return self._simulate_search(query)

    def _google_search(self, query: str, max_results: int) -> str:
        try:
            cmd = ["google", "--count", str(max_results), "--json", query]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    results = []
                    for item in data.get("results", [])[:max_results]:
                        results.append(
                            f"- {item.get('title', 'No title')}: {item.get('link', 'No link')}"
                        )
                    return f"Search results for '{query}':\n" + "\n".join(results)
                except json.JSONDecodeError:
                    return "Search completed but results could not be parsed."
            else:
                return self._simulate_search(query)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            return self._simulate_search(query)

    def _simulate_search(self, query: str) -> str:
        return f"""Search results for '{query}':
        
This is a simulated search result. To enable real web search, please install either:
- macOS mdfind (built-in on macOS)
- Google CLI tool: pip install google-cli

For now, consider researching '{query}' manually through your preferred search engine."""


class SummarizeTool:
    def __init__(self) -> None:
        self.max_summary_length = 200

    def execute(self, input_data: str, context: str | None = None) -> str:
        if not input_data or len(input_data.strip()) < 10:
            return "Insufficient content to summarize."

        try:
            if Path(input_data).exists():
                content = Path(input_data).read_text(encoding="utf-8")
            else:
                content = input_data

            summary = self._create_summary(content, context)
            return self._save_summary(summary, input_data)
        except Exception as e:
            return f"Summarization failed: {str(e)}"

    def _create_summary(self, content: str, context: str | None = None) -> str:
        lines = content.strip().split("\n")

        # Extract key sentences or paragraphs
        key_points = []
        for line in lines:
            line = line.strip()
            if len(line) > 20 and not line.startswith("#"):
                # Simple heuristic: longer sentences with important keywords
                if any(
                    keyword in line.lower()
                    for keyword in [
                        "important",
                        "key",
                        "main",
                        "result",
                        "conclusion",
                        "summary",
                    ]
                ):
                    key_points.append(line)
                elif len(key_points) < 3 and len(line) > 50:
                    key_points.append(line)

        if not key_points:
            # Fall back to first few lines
            key_points = [line for line in lines[:3] if len(line.strip()) > 20]

        summary = "\n".join(key_points[:5])

        if len(summary) > self.max_summary_length:
            summary = summary[: self.max_summary_length].rsplit(".", 1)[0] + "."

        context_part = f"\nContext: {context}" if context else ""
        return f"Summary{context_part}:\n{summary}"

    def _save_summary(self, summary: str, source: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = Path("storage/summaries") / f"summary_{timestamp}.txt"
        summary_file.parent.mkdir(exist_ok=True)

        summary_with_metadata = f"""Summary generated: {datetime.now().isoformat()}
Source: {source}
{'='*50}
{summary}
"""

        summary_file.write_text(summary_with_metadata, encoding="utf-8")
        return f"Summary saved to {summary_file}\n\n{summary}"


class NoteTool:
    def __init__(self) -> None:
        self.notes_dir = Path("storage/notes")
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self, content: str, note_type: str = "general", tags: list[str] | None = None
    ) -> str:
        if not content or len(content.strip()) < 5:
            return "Content too short to save as a note."

        try:
            note_id = self._save_note(content, note_type, tags or [])
            return f"Note saved with ID: {note_id}\nContent: {content[:100]}..."
        except Exception as e:
            return f"Failed to save note: {str(e)}"

    def _save_note(self, content: str, note_type: str, tags: list[str]) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        note_id = f"{note_type}_{timestamp}"
        note_file = self.notes_dir / f"{note_id}.md"

        note_metadata = {
            "id": note_id,
            "type": note_type,
            "created": datetime.now().isoformat(),
            "tags": tags,
        }

        note_content = f"""---
{json.dumps(note_metadata, indent=2)}
---

{content}
"""

        note_file.write_text(note_content, encoding="utf-8")
        return note_id

    def list_notes(self, note_type: str | None = None, limit: int = 10) -> str:
        try:
            notes = []
            for note_file in sorted(self.notes_dir.glob("*.md"), reverse=True)[:limit]:
                if note_type is None or note_file.name.startswith(f"{note_type}_"):
                    try:
                        content = note_file.read_text(encoding="utf-8")
                        # Extract metadata
                        if content.startswith("---\n"):
                            parts = content.split("---\n", 2)
                            if len(parts) >= 3:
                                metadata = json.loads(parts[1])
                                notes.append(
                                    f"- {metadata['id']}: {metadata.get('type', 'unknown')} {metadata['created']}"
                                )
                    except (json.JSONDecodeError, IndexError):
                        continue

            if notes:
                return "Recent notes:\n" + "\n".join(notes)
            else:
                return "No notes found."
        except Exception as e:
            return f"Failed to list notes: {str(e)}"
