"""CPU-only variant of config.py.

Kept as a separate module because several scripts and smoke_test.sh import it
by name. The GPU choice is the only difference, so it sets the env var and
hands off rather than repeating the session code.
"""

import os

os.environ["FOSAE_GPU"] = "0"

from config import load_session, clear_session, reload_session  # noqa: F401,E402
