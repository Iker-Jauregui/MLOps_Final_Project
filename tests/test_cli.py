"""
Integration testing with the CLI
"""
import pytest
from pathlib import Path
from click.testing import CliRunner
from cli.cli import cli


# Fixture
@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance for all CLI tests."""
    return CliRunner()


@pytest.fixture
def sample_image_path():
    """Provide path to test image."""
    return str(Path(__file__).parent / "sample.jpg")


def test_help(runner):
    """Tests the command-line interface help message."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output


# Testing of the predict_cli of the inference group
def test_predict_cli(runner, sample_image_path):
    """Tests the command-line interface predict command."""
    result = runner.invoke(cli, ["inference", "predict", sample_image_path])
    assert result.exit_code == 0
    assert "Predicted class:" in result.output


def test_predict_cli_with_custom_classes(runner, sample_image_path):
    """Tests predict with custom class names."""
    result = runner.invoke(
        cli,
        ["inference", "predict", sample_image_path, "--class-names", "cat,dog"],
    )
    assert result.exit_code == 0
    assert "Predicted class:" in result.output


# Testing of the resize_cli of the transform group
def test_resize_cli(runner, sample_image_path):
    """Tests the command-line interface resize command."""
    result = runner.invoke(cli, ["transform", "resize", sample_image_path, "32", "32"])
    assert result.exit_code == 0
    assert "32" in result.output


def test_resize_cli_invalid_width(runner, sample_image_path):
    """Tests resize with invalid width."""
    result = runner.invoke(cli, ["transform", "resize", sample_image_path, "0", "32"])
    assert result.exit_code == 0
    assert "Error" in result.output