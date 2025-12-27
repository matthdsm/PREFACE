"""
PREFACE CLI entry point.
"""

import typer

from preface import __version__
from preface.predict import preface_predict
from preface.train import preface_train
from preface.utils.ffy import wisecondorx_ffy
from preface.utils.npz_to_parquet import npz_to_parquet

# Version
VERSION: str = __version__

# Initialize Typer app
app = typer.Typer(help="PREFACE - PREdict FetAl ComponEnt")
app.command(name="predict")(preface_predict)
app.command(name="train")(preface_train)
app.command(name="version")(lambda: typer.echo(f"PREFACE version {VERSION}"))

# Utilities group
utils_app = typer.Typer(help="Utility scripts")
utils_app.command(name="npz-to-parquet")(npz_to_parquet)
utils_app.command(name="ffy")(wisecondorx_ffy)
app.add_typer(utils_app, name="utils")

if __name__ == "__main__":
    app()
