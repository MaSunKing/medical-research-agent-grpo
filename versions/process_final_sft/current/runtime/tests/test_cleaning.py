import copy
import unittest

from trainable_agent.cleaning import clean_chunk, clean_observation
from trainable_agent.tools import normalize


def chunk(text):
    return dict(source_id="PMID:1", chunk_id="PMID:1#c1", title="Study", text=text,
                metadata={"authors": ["Author"], "year": 2024, "url": "https://example.invalid/source"})


class CleanupTests(unittest.TestCase):
    def test_mixed_ui_removed_medical_text_and_identity_preserved(self):
        body = "In adults, outcome improved by 12 m; no safety benefit was established.\n"
        original = chunk("Accept all cookies\n" + body + "Advertisement\n")
        saved = copy.deepcopy(original)
        out = clean_chunk(original)
        self.assertEqual(out["text"], body)
        for key in ("source_id", "chunk_id", "title", "metadata"):
            self.assertEqual(out[key], original[key])
        self.assertEqual(original, saved)
        self.assertEqual(len(out["cleaning"]["removed_spans"]), 2)
        self.assertEqual("".join(original["text"][a:b] for a, b in out["cleaning"]["kept_spans"]), body)

    def test_keywords_inside_medical_prose_are_not_deleted(self):
        text = "Patients log in to continue treatment monitoring. Cookie exposure was not measured.\nReferences\nSafety remains uncertain."
        self.assertEqual(clean_chunk(chunk(text))["text"], text)

    def test_negative_conflicting_repeated_results_preserved(self):
        text = "No benefit was observed in adults.\nNo benefit was observed in adults.\nAnother trial reported benefit."
        self.assertEqual(clean_chunk(chunk(text))["text"], text)

    def test_uncertain_navigation_and_html_preserved(self):
        for text in ("Home treatment improves mobility.", "Menu: dosage, adverse effects, contraindications", "<p>Risk was 3%.</p>"):
            self.assertEqual(clean_chunk(chunk(text))["text"], text)

    def test_only_known_ui_becomes_source_unreadable(self):
        obs = dict(status="ok", error_code="", retryable=False, candidates=[],
                   chunks=[chunk("Home | About us | Contact\nSign in to continue\n")])
        out, audit = clean_observation(obs)
        self.assertEqual(out["status"], "access_blocked")
        self.assertEqual(out["chunks"], [])
        self.assertFalse(audit["medical_relevance_assessed"])
        self.assertEqual(len(obs["chunks"]), 1)

    def test_irrelevant_body_is_not_programmatically_removed(self):
        text = "A study of a different condition reported results in children."
        out, _ = clean_observation(dict(status="ok", candidates=[], chunks=[chunk(text)]))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["chunks"][0]["text"], text)

    def test_backend_original_and_metadata_survive_normalization(self):
        raw = {"data": [{"source_id": "PMID:1#c1", "text": "No effect.", "authors": ["A"],
                         "doi": "example", "year": 2024}], "extra_source_record": {"date": "2024"}}
        out = normalize(raw, "paper", "PMID:1")
        self.assertEqual(out["backend_return"], raw)
        self.assertEqual(out["chunks"][0]["metadata"]["authors"], ["A"])
        out["backend_return"]["data"][0]["text"] = "Changed copy"
        self.assertEqual(raw["data"][0]["text"], "No effect.")

    def test_failures_not_reclassified(self):
        obs = dict(status="environment_failure", error_code="transport", chunks=[], candidates=[])
        out, _ = clean_observation(obs)
        self.assertEqual(out, obs)

    def test_contiguous_span_mapping_is_compact(self):
        out = clean_chunk(chunk("\n".join(f"Clinical result {i}." for i in range(200))))
        self.assertEqual(len(out["cleaning"]["kept_spans"]), 1)

    def test_body_with_access_notice_is_not_exempt(self):
        out, _ = clean_observation(dict(status="ok", candidates=[],
                          chunks=[chunk("Sign in to continue\nRisk was 3% in adults.\n")]))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["chunks"][0]["text"], "Risk was 3% in adults.\n")

    def test_navigation_only_is_not_exempt(self):
        out, _ = clean_observation(dict(status="ok", candidates=[],
                          chunks=[chunk("Home | About us | Contact\nAccept all cookies\n")]))
        self.assertEqual(out["status"], "source_unreadable")

    def test_only_explicit_backend_access_error_is_classified(self):
        blocked = normalize({"failed": True, "error_code": "captcha_required"}, "paper", "PMID:1")
        unknown = normalize({"failed": True, "error": "cannot open"}, "paper", "PMID:1")
        self.assertEqual(blocked["status"], "access_blocked")
        self.assertEqual(unknown["status"], "source_unreadable")

    def test_configuration_failures_stop_not_exempt(self):
        with self.assertRaisesRegex(RuntimeError, "requires_attention"):
            normalize({"failed": True, "error_code": "quota_exhausted"}, "web")
