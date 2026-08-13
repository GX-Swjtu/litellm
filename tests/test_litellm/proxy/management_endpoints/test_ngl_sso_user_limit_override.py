import os
from unittest.mock import patch

import pytest

from litellm.proxy.management_endpoints.ui_sso import _raise_if_sso_exceeds_free_user_limit


@pytest.mark.asyncio
async def test_platform_override_disables_sso_user_limit():
    with patch.dict(os.environ, {"NGL_LITELLM_UNLIMITED_SSO_USERS": "true"}):
        await _raise_if_sso_exceeds_free_user_limit(premium_user=False, prisma_client=None)
