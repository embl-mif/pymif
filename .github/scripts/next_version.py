"""Compute the next zarrsmith release version for a merged pull request.

Usage: python next_version.py <pr.json>

``pr.json`` is the output of ``gh pr view <n> --json labels``. Existing tags are
read from the local git repository. Prints the next version (``X.Y.Z``,
without the leading ``v``), or nothing when the PR should not be released.

Releases are opt-in: a PR is released only if it carries one of the labels
``release:major``, ``release:minor`` or ``release:patch``. Without such a label
nothing is tagged. If several are present the largest bump wins.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

# Tags with a major at or above this are legacy calendar versions and are ignored.
CALVER_MAJOR = 2000

TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
KINDS = ("major", "minor", "patch")  # ordered by priority


def latest_version(tags: list[str]) -> tuple[int, int, int] | None:
    versions = []
    for tag in tags:
        m = TAG_RE.match(tag.strip())
        if m:
            v = tuple(int(x) for x in m.groups())
            if v[0] < CALVER_MAJOR:
                versions.append(v)
    return max(versions) if versions else None


def bump_kind(pr: dict) -> str | None:
    labels = {label["name"].lower() for label in pr.get("labels") or []}
    for kind in KINDS:
        if f"release:{kind}" in labels:
            return kind
    return None


def next_version(tags: list[str], pr: dict) -> str | None:
    kind = bump_kind(pr)
    if kind is None:
        return None
    major, minor, patch = latest_version(tags) or (0, 0, 0)
    if kind == "major":
        return f"{major + 1}.0.0"
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        pr = json.load(fh)
    tags = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    print(next_version(tags, pr) or "")


if __name__ == "__main__":
    main()
