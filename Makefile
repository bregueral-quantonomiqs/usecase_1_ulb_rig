.PHONY: clean clean-all fmt test

clean:
	@bash scripts/clean.sh

# Removes .venv too (use with care)
clean-all:
	@bash scripts/clean.sh
	@rm -rf .venv

fmt:
	@black . || true
	@isort . || true

test:
	@pytest -q || true

.PHONY: demo
demo:
	@echo "Starting Streamlit demo..."
	@which streamlit >/dev/null 2>&1 && streamlit run app/streamlit_app.py || ./.venv/bin/streamlit run app/streamlit_app.py
