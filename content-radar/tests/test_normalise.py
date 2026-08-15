from radar.normalise import canonical_url, normalise_title, url_hash


class TestCanonicalUrl:
    def test_strips_tracking_params(self):
        messy = "https://example.com/post?utm_source=x&utm_medium=y&id=7&ref=hn"
        assert canonical_url(messy) == "https://example.com/post?id=7"

    def test_drops_www_and_lowercases_host(self):
        assert canonical_url("HTTPS://WWW.Example.com/Post") == "https://example.com/Post"

    def test_drops_fragment_and_trailing_slash(self):
        assert canonical_url("https://example.com/post/#section") == "https://example.com/post"

    def test_bare_root_keeps_its_slash(self):
        assert canonical_url("https://example.com/") == "https://example.com/"

    def test_query_order_does_not_matter(self):
        assert canonical_url("https://e.com/p?b=2&a=1") == canonical_url("https://e.com/p?a=1&b=2")

    def test_default_port_is_dropped(self):
        assert canonical_url("https://example.com:443/p") == "https://example.com/p"

    def test_empty_input_is_safe(self):
        assert canonical_url("") == ""

    def test_the_same_article_shared_three_ways_collapses(self):
        variants = [
            "https://www.example.com/ai-act-guide?utm_campaign=newsletter",
            "http://example.com/ai-act-guide/",
            "https://example.com/ai-act-guide#intro",
        ]
        hashes = {url_hash(v) for v in variants}
        # http vs https is a real difference; www, tracking and fragments are not.
        assert len(hashes) == 2


class TestNormaliseTitle:
    def test_strips_editorial_prefix(self):
        assert normalise_title("Exclusive: OpenAI ships agents").startswith("openai")

    def test_strips_stacked_prefixes(self):
        assert normalise_title("Breaking: Exclusive: Model released").startswith("model")

    def test_inflections_fold_together(self):
        # The point of stemming: tense and number must not split a story.
        assert normalise_title("OpenAI ships agents") == normalise_title("OpenAI shipped agent")
        assert normalise_title("Model released") == normalise_title("Model releases")

    def test_strips_publication_suffix(self):
        title = "Regulators publish new AI guidance for banks | Financial Times"
        assert "financial times" not in normalise_title(title)

    def test_keeps_short_titles_intact(self):
        # Suffix stripping must not eat a genuinely short headline.
        assert normalise_title("AI - now") != ""

    def test_drops_stopwords_and_punctuation(self):
        assert normalise_title("The Rise of the Agent!") == "rise agent"

    def test_reordered_headline_normalises_comparably(self):
        a = set(normalise_title("OpenAI ships new agent framework").split())
        b = set(normalise_title("New agent framework shipped by OpenAI").split())
        assert {"openai", "agent", "framework"} <= a & b

    def test_empty_input_is_safe(self):
        assert normalise_title("") == ""
