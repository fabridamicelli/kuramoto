test:
	pytest tests/test_initialization.py

release:
	rm dist/*
	python setup.py sdist bdist_wheel
	twine upload dist/*
