"""Grammars shared by the fakes, copied from the siblings' interfaces (see FIDELITY.md).

``tests/test_fakebin_grammars.py`` cross-checks the ones that also exist as patterns in the
pinned sibling schemas.
"""

LABEL = r"[a-z0-9][a-z0-9_.-]{0,63}=[A-Za-z0-9._:/#@+=,-]{0,256}"
ACTOR = r"(human|auto):[A-Za-z0-9._@-]{1,64}"
DECISION_ID = r"[A-Za-z0-9._:@/-]{1,128}"
CS_ID = r"cs-[a-z2-7]{8}"
TC_ID = r"tc-[a-z2-7]{8}"
WT_ID = r"wt-[a-z2-7]{8}"
KIND = r"[a-z0-9][a-z0-9_-]{0,63}"
SKILL = r"[A-Za-z0-9][A-Za-z0-9:._-]{0,127}"
# claude-session durations: seconds as a number, or number + ms|s|m|h|d.
CS_DURATION = r"[0-9]+(\.[0-9]+)?(ms|s|m|h|d)?"
# termctl durations: integer + ms|s|m, bare integer = seconds, at most 10m.
TC_DURATION = (
    r"((0|[1-9][0-9]{0,4}|[1-5][0-9]{5}|600000)ms|(0|[1-9][0-9]?|[1-5][0-9]{2}|600)s?|([0-9]|10)m)"
)
WT_DURATION = r"[1-9][0-9]*[smhdw]"
HOLDER = r"[A-Za-z0-9._:/#@+=,-]{1,64}"
WT_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
SESSION_NAME = r"[A-Za-z0-9_-]{1,64}"
ENV_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
PLACEMENT = r"[a-z0-9][a-z0-9_-]{0,63}"
REPO_ID = (
    r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)*(?::[0-9]{1,5})?"
    r"(?:/(?!\.\.?(?:/|$))[A-Za-z0-9._-]+)+"
)
SHA = r"[0-9a-f]{40}"
