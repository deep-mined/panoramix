deps-pre:
	pip install pip-tools

deps-compile:
	pip-compile

install:
	pip-sync

notebook:
	jupyter notebook

app:
	streamlist run app.py
