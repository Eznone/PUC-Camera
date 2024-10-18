# Setting up a Python Virtual Environment (venv) on macOS and Windows

This guide will walk you through the steps of setting up a Python virtual environment using `venv` on both macOS and Windows. Virtual environments allow you to manage dependencies for your projects in an isolated environment.

## Prerequisites
- Ensure that Python is installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).
- To check if Python is installed, open a terminal (macOS) or command prompt (Windows) and type:
    ```bash
    Windows: python --version
    MacOS: python3 --version

## Installing Venv
    Windows: python -m venv .venv
    MacOS: python3 -m venv .venv

## Activating Virtual Environment
    Windows: .venv\Scripts\Activate
    MacOS: source .venv/bin/activate

## Installing Packages
    Windows: pip install -r requirements.txt
    MacOS: pip3 install -r requirements.txt

## Deactivating
    Windwos: deactivate
    MacOS: deactivate

## Solution to venv: command not found
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    