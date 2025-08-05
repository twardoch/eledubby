# this_file: utils/progress.py
"""Progress tracking utilities."""

from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)


class ProgressTracker:
    """Unified progress tracking for the application."""

    def __init__(self):
        """Initialize progress tracker."""
        self.console = Console()

    @contextmanager
    def track_segments(self, total: int, description: str = "Processing segments"):
        """Track progress for segment processing.

        Args:
            total: Total number of segments
            description: Task description

        Yields:
            Progress update function
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(description, total=total)

            def update(advance: int = 1, description: str | None = None):
                if description:
                    progress.update(task, description=description)
                progress.advance(task, advance)

            yield update

    @contextmanager
    def track_file_operation(self, description: str):
        """Track a file operation with spinner.

        Args:
            description: Operation description

        Yields:
            Progress context
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(description)
            yield progress
            progress.update(task, completed=True)

    def print_summary(self, stats: dict):
        """Print processing summary.

        Args:
            stats: Dictionary of statistics to display
        """
        self.console.print("\n[bold green]Processing Summary:[/bold green]")
        for key, value in stats.items():
            self.console.print(f"  [cyan]{key}:[/cyan] {value}")
