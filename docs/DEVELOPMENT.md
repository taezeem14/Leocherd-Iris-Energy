# Development & Contributing Guide

## Code Standards
- **Python Version Target:** Python 3.11+
- **Linter & Formatter:** `ruff check .` and `ruff format .`
- **Typing:** Strict typing annotations across all public functions and data models using `pydantic` and `typing`.
- **Testing:** Unit tests in `tests/unit/`, integration pipelines in `tests/integration/`.

## Running the API Locally
```bash
python main.py run-server --host 127.0.0.1 --port 8000
```
API Documentation will be live at `http://127.0.0.1:8000/docs`.
