"""
Static checks on docker/Dockerfile.non_root.

The non_root image is intended for deployment into hardened Kubernetes
clusters where `securityContext.runAsNonRoot: true` is enforced. The
kubelet validates non-root status by parsing the image's USER field as
an integer — a string name like "nobody" is rejected with
CreateContainerConfigError because the kubelet cannot resolve
/etc/passwd inside the image at admission time.
"""

import os
import re
from typing import Final

import pytest

DOCKERFILE_PATH: Final = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "docker",
    "Dockerfile.non_root",
)


def _final_user_directive(dockerfile_text: str) -> str:
    """Return the value of the last `USER` directive in the file."""
    matches = re.findall(r"^USER\s+(\S+)\s*$", dockerfile_text, re.MULTILINE)
    assert matches, "Dockerfile.non_root has no USER directive"
    return matches[-1]


def _dockerfile_text() -> str:
    with open(DOCKERFILE_PATH, "r", encoding="utf-8") as dockerfile:
        return dockerfile.read()


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile.non_root not present in this checkout",
)
def test_final_user_directive_is_numeric():
    """The runtime USER must be a numeric UID so kubelet's runAsNonRoot
    admission check (strconv.Atoi) succeeds."""
    with open(DOCKERFILE_PATH, "r", encoding="utf-8") as f:
        contents = f.read()

    final_user = _final_user_directive(contents)

    assert final_user.isdigit(), (
        f"Dockerfile.non_root final USER is {final_user!r}; must be a numeric UID "
        "so Kubernetes' runAsNonRoot admission check can verify non-root status. "
        "See https://kubernetes.io/docs/tasks/configure-pod-container/security-context/"
    )

    assert int(final_user) != 0, (
        f"Dockerfile.non_root final USER is {final_user} (root); the non_root image "
        "must run as a non-zero UID."
    )


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile.non_root not present in this checkout",
)
def test_apk_downloads_use_architecture_scoped_shared_cache():
    contents: Final = _dockerfile_text()
    cache_mount: Final = (
        "--mount=type=cache,id=litellm-apk-${TARGETARCH},"
        "target=/var/cache/apk,sharing=locked"
    )

    assert contents.count("ARG TARGETARCH") == 2
    assert contents.count(cache_mount) == 2
    assert contents.count("ln -s /var/cache/apk /etc/apk/cache") == 2
    assert contents.count("unlink /etc/apk/cache") == 2
    assert "apk add --no-cache" not in contents
    assert "apk upgrade --no-cache" not in contents


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile.non_root not present in this checkout",
)
def test_uv_uses_unversioned_system_python_without_downloading_another_version():
    contents: Final = _dockerfile_text()

    assert "python3-dev" in contents
    assert "apk add python3 bash" in contents
    assert "UV_PYTHON_DOWNLOADS=0" in contents
    assert contents.count("--python python3") == 3
    assert re.search(r"\bpython-\d+\.\d+(?:-dev)?\b", contents) is None
    assert re.search(r"--python python\d+\.\d+", contents) is None


@pytest.mark.skipif(
    not os.path.exists(DOCKERFILE_PATH),
    reason="Dockerfile.non_root not present in this checkout",
)
def test_apk_retries_propagate_the_last_failure():
    contents: Final = _dockerfile_text()

    assert contents.count('echo "apk add failed after 3 retries" >&2; exit 1') == 2
    assert contents.count('echo "apk upgrade failed after 3 retries" >&2; exit 1') == 1
