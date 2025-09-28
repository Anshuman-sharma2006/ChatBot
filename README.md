# Project Setup with UV

This guide explains how to set up and manage your Python workspace using `uv`, including installing dependencies.

---

## 1. Install `uv`

First, navigate to your project folder (workspace) and install `uv`:

pip install uv

## 2. Verify Installation

uv --version


## 3. Initialize the Workspace

Initialize your current folder to work with uv:

uv init .

## 4. Add Dependencies

If you have a requirements.txt file containing all the required Python libraries, add them using:

uv add -r req.txt.



## 5. If You want To Run Any command You Should use:
uv run <file_name>.<br>
uv  run streamlit run cognibot.py(This is for The Streamlit).


