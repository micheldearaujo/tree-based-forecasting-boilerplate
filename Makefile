install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

dataset:
	python -m src/data/make_dataset.py
	python -m src/features/build_features.py

test:
	python -m pytest -vv --cov=src/utils tests/test_utils.py

lint:
	pylint --disable=R,C *.py
	
format:
	black *.py

clean:
	rm -rf __pycache__
	rm -f *.log
	rm -f *.log
	rm -f .coverage*
