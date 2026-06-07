# my-portfolio-notebooks

Notebooks and utilities used to support portfolio analysis and trading activities.

## Requirements

- Python 3.14
- `uv` (Python package and virtual environment manager)

## Install `uv`

### macOS

Using the official installer script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Using Homebrew:

```bash
brew install uv
```

### Windows

Using PowerShell (official installer):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Using Chocolatey:

```powershell
choco install uv
```

### Linux

`apt` is not an official `uv` installation method in the upstream documentation.

Using the official installer script:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create and initialize the virtual environment (Python 3.14)

From the project root, run:

```bash
uv venv --python 3.14
```

This creates a `.venv` directory.

### Activate the virtual environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```cmd
.venv\Scripts\activate.bat
```

## Install dependencies

With the environment activated:

```bash
uv pip install -e .
```

## Run Jupyter Lab

```bash
jupyter lab
```

## Verify Python version

```bash
python --version
```

Expected output should show Python 3.14.x.
