"""
PREFACE CLI entry point.
"""

import os
import warnings
import logging
from rich.logging import RichHandler

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning, module="keras")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import typer  # noqa: E402

from preface import __version__  # noqa: E402
from preface.predict import preface_predict  # noqa: E402
from preface.train import preface_train  # noqa: E402
from preface.utils.ffy import wisecondorx_ffy  # noqa: E402
from preface.utils.npz_to_parquet import npz_to_parquet  # noqa: E402

# Configure logging
logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_suppress=[typer])],
)

# Version
VERSION: str = __version__

# Initialize Typer app
app = typer.Typer(help="PREFACE - PREdict FetAl ComponEnt")
app.command(name="predict")(preface_predict)
app.command(name="train")(preface_train)
app.command(name="version")(lambda: print(f"PREFACE version {VERSION}"))

# Utilities group
utils_app = typer.Typer(help="Utility scripts")
utils_app.command(name="npz-to-parquet")(npz_to_parquet)
utils_app.command(name="ffy")(wisecondorx_ffy)
app.add_typer(utils_app, name="utils")

if __name__ == "__main__":
    app()
