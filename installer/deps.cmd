@echo off
rem Run by setup. Installs Python and the app's dependencies inside this folder, so
rem uninstalling removes them too. Extra arguments (setup passes --extra cuda on an
rem NVIDIA PC) go through to uv.
set "UV_PYTHON_INSTALL_DIR=%~dp0python"
set "UV_CACHE_DIR=%~dp0uv-cache"
rem Never bind the app to a Python the user installed and may later remove.
set "UV_PYTHON_PREFERENCE=only-managed"
"%~dp0uv.exe" sync --frozen --project "%~dp0." %*
