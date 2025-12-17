"""
Main CLI or app entry point
"""

import click
from mylib.classifier import predict, resize


# We create a group of commands
@click.group()
def cli():
    """Main CLI to perform image operations."""


# ============================================================================
# INFERENCE GROUP - Image inference operations
# ============================================================================
@cli.group()
def inference():
    """Commands for image inference operations."""


# We create a command, named predict, associated with the previous group
@inference.command("predict")
@click.argument("image-path", type=str)
@click.option(
    "--class-names",
    default="cardboard,paper,plastic,metal,trash,glass",
    type=str,
    help="Comma-separated class names (e.g., 'cat,dog,bird').",
)
def predict_cli(image_path, class_names):
    """Predict image class.

    Example:
        uv run python -m cli.cli inference predict 'sample.jpg'
    """
    try:
        # Convert comma-separated string to list
        class_list = [c.strip() for c in class_names.split(",")]
        result = predict(image_path, class_list)
        click.echo(click.style(f"Predicted class: {result}", fg="green"))
    except (FileNotFoundError, IOError, ValueError) as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)


# ============================================================================
# TRANSFORM GROUP - Image transform operations
# ============================================================================


@cli.group()
def transform():
    """Commands for image transform operations."""


@transform.command("resize")
@click.argument("image-path", type=str)
@click.argument("width", type=int)
@click.argument("height", type=int)
def resize_cli(image_path, width, height):
    """Resize image.

    Example:
        uv run python -m cli.cli transform resize 'sample.jpg' 224 224
    """
    try:
        if width <= 0:
            raise ValueError("'width' must be a positive value")
        if height <= 0:
            raise ValueError("'height' must be a positive value")

        result = resize(image_path, width, height)
        click.echo(click.style(f"Resized to: {result}", fg="green"))
    except (FileNotFoundError, IOError, ValueError) as e:
        click.echo(click.style(f"Error: {str(e)}", fg="red"), err=True)


# Main entry point
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli()
