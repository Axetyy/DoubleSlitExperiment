"""Application entry point for the double-slit simulation."""

from controller import SimulationManager
from ui.matplotlib_ui import MatplotlibUI


def main():
    """Start the default Matplotlib presentation."""
    MatplotlibUI(SimulationManager()).show()


if __name__ == "__main__":
    main()
