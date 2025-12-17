import typer

from preface.predict import preface_predict
from preface.train import preface_train
from preface.utils.npz_to_parquet import npz_to_parquet
from preface.utils.ffy import ffy
from preface import __version__

# Version
VERSION: str = __version__

# Initialize Typer app
app = typer.Typer(help="PREFACE - PREdict FetAl ComponEnt")
app.command(name="predict")(preface_predict)
app.command(name="train")(preface_train)

# Utilities group
utils_app = typer.Typer(help="Utility scripts")
utils_app.command(name="npz-to-parquet")(npz_to_parquet)
utils_app.command(name="ffy")(ffy)
app.add_typer(utils_app, name="utils")

if __name__ == "__main__":
    app()
