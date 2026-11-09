"""aioice workaround: don't wipe ICE components when nothing resolved.

Why this exists
---------------
Firefox publishes local host candidates as mDNS names (<uuid>.local).
aioice resolves them on receipt; if resolution fails (multicast broken,
VPN/TUN interference, headless environments...) and the remote then signals
end-of-candidates, ``Connection._prune_components()`` empties the component
set because *no* remote candidate resolved. aioice then gathers **no local
candidates**, our SDP answer carries none, and the peer can never reach us:

    Firefox: connectionState=failed, zero inbound RTP, video stays blank
    (Chrome still works: it tolerates missing sprop-parameter-sets and
    resolves candidate exchange differently, so it never hit this path).

Guarding the prune so it only runs when at least one remote candidate
resolved lets us keep publishing our own candidates. The peer's checks then
arrive as peer-reflexive candidates and ICE completes normally — verified
end-to-end with desktop Firefox (Selenium) after this patch.

Source: aioice/ice.py ``Connection._prune_components``.
Idempotent: safe to import from multiple places.
"""

import aioice.ice as _ice

_orig_prune_components = _ice.Connection._prune_components


def _prune_components(self):
    if not self._remote_candidates:
        # Nothing resolved — keep self._components intact so local ICE
        # gathering still runs and our candidates reach the answer SDP.
        return
    return _orig_prune_components(self)


if not getattr(_ice.Connection._prune_components, "_livetalking_fix", False):
    _prune_components._livetalking_fix = True
    _prune_components.__doc__ = _orig_prune_components.__doc__
    _prune_components.__wrapped__ = _orig_prune_components
    _ice.Connection._prune_components = _prune_components