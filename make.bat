@echo off
rem Windows shim -- `make <target>` now routes to tasks.py.
rem In cmd, a .bat in the current folder runs before any make.exe on PATH.
uv run python tasks.py %*