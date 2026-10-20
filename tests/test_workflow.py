import bz2
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WikipediaWorkflowTest(unittest.TestCase):
    def test_offline_pipeline_uses_external_paths_and_preserves_category_provenance(self):
        text = ("D Stadt Basel isch e Stadt in dr Schwiiz. Si ligt am Rhy und het e langi "
                "Gschicht mit vilne Bruggene und alte Hüüser. D Lüt rede do Baseldytsch "
                "und bsueche gärn s Museum.")
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            extracted = data / "intermediate/extracted"
            extracted.mkdir(parents=True)
            (data / "raw").mkdir()
            (extracted / "als_dialect_map.json").write_text(json.dumps({"1": "Baseldytsch", "2": None}))
            pages = "".join(f"<page><title>{title}</title><ns>0</ns><id>{identifier}</id>"
                            f"<revision><text>{text}</text></revision></page>"
                            for identifier, title in [("1", "Basel"), ("2", "Fremd"), ("3", "Ohni Kategorie")])
            xml = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">' + pages + '</mediawiki>'
            (data / "raw/alswiki-latest-pages-articles.xml.bz2").write_bytes(bz2.compress(xml.encode()))
            env = {**os.environ, "WIKI_DATA_ROOT": str(data), "PYTHON": sys.executable}
            for key in ("OUT_DIR", "WIKI_ARTICLE_DUMP", "RUN_AUTHOR_EXPERIMENTS", "REFRESH_CATEGORIES"):
                env.pop(key, None)
            subprocess.run(["bash", str(ROOT / "run_pipeline.sh")], cwd=data, env=env,
                           check=True, capture_output=True, text=True)
            rows = [json.loads(line) for line in (data / "results/clean_translation_corpus.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in rows], ["1", "3"])
            self.assertEqual(rows[0]["dialect"], "Baseldytsch")
            self.assertEqual(rows[0]["dialect_label_source"], "category")
            self.assertIsNone(rows[1]["dialect_label_source"])
            self.assertIsNone(rows[1]["translation"])

    def test_cleaner_preserves_paragraph_boundaries(self):
        cleaner = load_module("wiki_cleaner", "author_dialect/export_clean_translation_corpus.py")
        text = "D Stadt isch am Rhy und het vil alti Hüüser.\n\nD Lüt bsueche s Museum."
        self.assertIn("\n\n", cleaner.clean_text(text))

    def test_segments_reconstruct_the_original_article(self):
        flatten = load_module("wiki_flatten", "OpenRouter_wikipedia_translate/scripts/flatten_segments_jsonl.py")
        article = {"id": "article-1", "title": "Beispiel", "segments": [
            {"gsw": "D Stadt isch gross.", "de": "Die Stadt ist gross."},
            {"gsw": "Si ligt am Rhy.", "de": "Sie liegt am Rhein."},
        ]}
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.jsonl", Path(folder) / "pairs.jsonl"
            source.write_text(json.dumps(article) + "\n")
            output.write_text("".join(json.dumps(row) + "\n" for row in flatten.flatten(article)))
            flatten.verify(source, output)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["pair_id"] for row in rows], ["article-1:0", "article-1:1"])


if __name__ == "__main__":
    unittest.main()
