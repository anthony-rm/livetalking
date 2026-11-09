"""aioice workaround #2: don't fail ICE immediately when the controlled side
has no pairs yet.

Why this exists
---------------
The server's gatherer is created with ``ice_controlling=False`` (see
``RTCIceGatherer.__init__``), so the server is the *controlled* side. Per
RFC 8445 7.2.1.3 the controlled side should wait for incoming checks and
reply, not initiate.

aioice's ``Connection.connect()`` loops ``check_periodic()`` and, when the
check list is empty, raises ``ICE_FAILED`` immediately:

    if self._check_list:
        res = await self._check_list_state.get()
    else:
        res = ICE_FAILED

On this machine the client's candidates are mDNS names (Firefox obfuscates
local host candidates as ``<uuid>.local``) and aioice cannot resolve them, so
the server has **zero** remote candidates when it enters ``connect()``. The
controlled side's check list is therefore empty, and it fails before the
client's peer-reflexive checks arrive — even though aioice *does* support
learning peer-reflexive candidates (``Connection.check_incoming``,
``Discovered peer reflexive candidate``).

Observed on the wire (aioice debug log):

    RTCIceTransport(controlled) - new -> checking
    RTCIceTransport(controlled) - checking -> failed
    ...
    Discovered peer reflexive candidate ... fd00:1234:ffff::10 61129 typ prflx

The peer-reflexive candidate was discovered *after* ICE had already failed.

The patch: when the controlled side has no pairs and the remote has not yet
signalled end-of-candidates, wait briefly for incoming checks instead of
failing immediately. This is a no-op for the controlling side and for the
normal (non-mDNS) path, where the check list is non-empty.

Source: aioice/ice.py ``Connection.connect``.
Idempotent: safe to import from multiple places.
"""

import asyncio
import aioice.ice as _ice

_orig_connect = _ice.Connection.connect


async def _patched_connect(self):
    # Only the controlled side can hit the empty-check-list path; the
    # controlling side always has pairs to probe.
    if not self.ice_controlling and not self._check_list:
        # Give incoming (peer-reflexive) checks a chance to arrive.
        try:
            await asyncio.wait_for(self._check_list_state.get(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        # If checks arrived in the meantime, the queue was drained and
        # pairs were formed — fall through to the normal loop.
        if not self._check_list:
            # Still nothing. If the remote hasn't signalled end of
            # candidates, keep waiting a little longer rather than failing.
            if not self._remote_candidates_end:
                try:
                    await asyncio.wait_for(self._check_list_state.get(),
                                           timeout=10.0)
                except asyncio.TimeoutError:
                    pass
    return await _orig_connect(self)


if not getattr(_ice.Connection.connect, "_livetalking_fix", False):
    _patched_connect._livetalking_fix = True
    _patched_connect.__doc__ = _orig_connect.__doc__
    _patched_connect.__wrapped__ = _orig_connect
    _ice.Connection.connect = _patched_connect