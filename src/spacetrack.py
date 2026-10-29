"""Space-Track.org access: login, rate-limited queries, and an account check.

Space-Track (run by the US Space Force) keeps the full orbit history of every
tracked object. It needs a free account: https://www.space-track.org/auth/createAccount

Put your login in a file called .env in the project folder (git ignores it):

    SPACETRACK_USER=you@example.com
    SPACETRACK_PASSWORD=your-password

Then check that it works:

    python src/spacetrack.py
"""

import time
from pathlib import Path

import requests

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
BASE_URL = "https://www.space-track.org"
# Space-Track allows fewer than 30 queries a minute and 300 an hour; 15 s apart stays under both.
SECONDS_BETWEEN_QUERIES = 15
HEADERS = {"User-Agent": "satellite-reentry-student-project"}


def read_credentials():
    if not ENV_FILE.exists():
        raise SystemExit(f"No .env file found at {ENV_FILE}. See the top of this file for what to put in it.")
    values = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    if not values.get("SPACETRACK_USER") or not values.get("SPACETRACK_PASSWORD"):
        raise SystemExit(".env needs both SPACETRACK_USER and SPACETRACK_PASSWORD lines.")
    return values["SPACETRACK_USER"], values["SPACETRACK_PASSWORD"]


class SpaceTrack:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._last_query = 0.0
        self.login()

    def login(self):
        user, password = read_credentials()
        response = self.session.post(f"{BASE_URL}/ajaxauth/login",
                                     data={"identity": user, "password": password}, timeout=60)
        response.raise_for_status()
        if "Failed" in response.text:
            raise SystemExit("Space-Track login failed: check the username and password in .env")

    def get(self, path):
        """GET basicspacedata/<path>, spaced out to respect Space-Track's rate limits."""
        for attempt in range(2):
            wait = self._last_query + SECONDS_BETWEEN_QUERIES - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last_query = time.time()
            response = self.session.get(f"{BASE_URL}/basicspacedata/{path}", timeout=600)
            if response.status_code == 401 and attempt == 0:  # session expired during a long run
                self.login()
                continue
            response.raise_for_status()
            return response.text

    def close(self):
        self.session.get(f"{BASE_URL}/ajaxauth/logout", timeout=60)


def main():
    st = SpaceTrack()
    print("Logged in to Space-Track.")
    try:
        print("\ngp_history fields:")
        print(st.get("modeldef/class/gp_history/format/json")[:3000])
        print("\nISS's three earliest orbits:")
        print(st.get("query/class/gp_history/NORAD_CAT_ID/25544/orderby/EPOCH asc/limit/3/format/csv"))
    finally:
        st.close()


if __name__ == "__main__":
    main()
