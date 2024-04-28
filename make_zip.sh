#!/bin/bash

mv ./src/main.py ./src/__main__.py
zip -r -j app.zip ./src/*.py
mv ./src/__main__.py ./src/main.py

