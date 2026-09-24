from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"

# Importing the environments registers the Gym IDs.  Keep this registration
# convenient for the simulator environment, but do not make the lightweight
# data/conversion utilities depend on SAPIEN: the OpenPI environment uses the
# same ``galaxea_sim.utils`` package when converting HDF5 data.
try:
    import galaxea_sim.envs as _registered_envs  # noqa: F401
except ModuleNotFoundError as error:
    if error.name not in {"sapien", "sapien.core"}:
        raise
