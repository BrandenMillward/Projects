"""URL and title normalisation.

Two stories about the same announcement almost never arrive with the same URL or
the same headline. Everything downstream — dedupe, clustering, the seen-set —
depends on collapsing those differences first.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Params that identify the referrer rather than the content. Dropping them means
# the same article shared from three places hashes to one URL.
TRACKING_PREFIXES = ("utm_", "mc_", "pk_", "hsa_")
TRACKING_PARAMS = {
    "ref",
    "ref_src",
    "referrer",
    "source",
    "fbclid",
    "gclid",
    "igshid",
    "cmpid",
    "at_medium",
    "at_campaign",
}

# Leading noise editors put in front of the actual headline.
TITLE_PREFIX = re.compile(
    r"^\s*(exclusive|breaking|opinion|analysis|updated|video|watch|live)\s*[:\-–—]\s*",
    re.IGNORECASE,
)
# Trailing " - Publication" / " | Publication" suffixes.
TITLE_SUFFIX = re.compile(r"\s*[|\-–—]\s*[^|\-–—]{2,40}\s*$")
NON_WORD = re.compile(r"[^\w\s]")
WHITESPACE = re.compile(r"\s+")

# Words carrying no discriminating signal when comparing two headlines.
STOPWORDS = frozenset(
    """a an the and or but of for to in on at by with from as is are was were be been
    its it this that these those new now how why what when will can could should""".split()
)


def canonical_url(url: str) -> str:
    """Return a comparable form of *url*.

    Lowercases scheme and host, drops ``www.``, strips tracking parameters,
    sorts what remains, removes fragments, and normalises the trailing slash.
    """
    if not url:
        return ""

    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    host = parts.netloc.lower()

    if host.startswith("www."):
        host = host[4:]
    # Drop a default port so example.com and example.com:443 agree.
    if host.endswith(":443") and scheme == "https":
        host = host[:-4]
    elif host.endswith(":80") and scheme == "http":
        host = host[:-3]

    kept = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if k.lower() not in TRACKING_PARAMS
        and not any(k.lower().startswith(p) for p in TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(kept))

    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")

    return urlunsplit((scheme, host, path, query, ""))


def url_hash(url: str) -> str:
    """Stable short hash of the canonical URL, used as the seen-set key."""
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:16]


def _stem(word: str) -> str:
    """Crudely fold inflections so headline verb tense stops mattering.

    Outlets covering one story routinely differ only in tense or number —
    "OpenAI ships X" against "X shipped by OpenAI" — and without this they
    compare as different words and fail to cluster. This is deliberately not a
    real stemmer: it only has to make two headlines about the same event agree.
    """
    if len(word) <= 3:
        return word

    for suffix, keep in (("ing", 4), ("ed", 3), ("es", 3)):
        if word.endswith(suffix) and len(word) > keep:
            word = word[: -len(suffix)]
            break
    else:
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]

    # "shipped" -> "shipp" -> "ship"
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "lsz":
        word = word[:-1]
    # "release" / "released" / "releases" all land on "releas"
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]

    return word


def normalise_title(title: str) -> str:
    """Reduce a headline to comparable tokens.

    Strips editorial prefixes and publication suffixes, drops punctuation and
    stopwords, and lowercases. The result is only ever compared, never shown.
    """
    if not title:
        return ""

    text = title.strip()
    # Prefixes can stack ("Exclusive: Breaking: ...").
    while True:
        stripped = TITLE_PREFIX.sub("", text)
        if stripped == text:
            break
        text = stripped

    # Only strip a suffix when something substantial survives it.
    candidate = TITLE_SUFFIX.sub("", text)
    if len(candidate) >= 20:
        text = candidate

    text = NON_WORD.sub(" ", text.lower())
    tokens = [_stem(t) for t in WHITESPACE.split(text) if t and t not in STOPWORDS]
    return " ".join(t for t in tokens if t)
