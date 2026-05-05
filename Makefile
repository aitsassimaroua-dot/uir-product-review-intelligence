.PHONY: install train eval demo ui test clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m src.model.train

eval:
	python -m src.model.evaluate

demo:
	python -m src.main --product "Wireless Earbuds X" --reviews data/processed/sample_reviews.csv

ui:
	streamlit run streamlit_app.py

test:
	pytest -q

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
