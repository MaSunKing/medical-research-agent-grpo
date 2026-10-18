import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from worker import credentials

class Contracts(unittest.TestCase):
    def test_aliases_and_quotes(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True):
            path=Path(d)/'test.env'
            path.write_text("PUBMED_API_KEY='fake-ncbi'\nSEMANTIC_SCHOLAR_API_KEY=\"fake-s2\"\nMINERU_API_KEY='fake-pdf'\n")
            path.chmod(0o600)
            values=credentials(path)
            self.assertEqual(values,dict(NCBI_API_KEY='fake-ncbi',S2_API_KEY='fake-s2',MINERU_API_TOKEN='fake-pdf'))
    def test_conflicting_alias_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True):
            path=Path(d)/'test.env';path.write_text('NCBI_API_KEY=a\nPUBMED_API_KEY=b\n');path.chmod(0o600)
            with self.assertRaises(ValueError):credentials(path)

if __name__=='__main__':unittest.main()
