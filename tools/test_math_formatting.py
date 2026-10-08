"""Guard Markdown math fences and TeX structure; not a MathJax renderer."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    'README.md',
    'versions/single_lora/README.md',
    'versions/dual_lora/README.md',
    'versions/process_final_sft/README.md',
    'versions/process_final_sft/grpo.md',
)


class MathFormattingTests(unittest.TestCase):
    def test_math_uses_fences_and_balanced_tex(self):
        total = 0
        for name in DOCS:
            text = (ROOT / name).read_text(encoding='utf-8')
            with self.subTest(document=name):
                self.assertNotRegex(text, r'(?m)^\s*\$\$\s*$')
                blocks = re.findall(r'(?m)^```math\n([\s\S]*?)^```\s*$', text)
                self.assertEqual(len(blocks), text.count('```math'))
                self.assertTrue(blocks)
                for block in blocks:
                    depth = 0
                    # Escaped braces are literal delimiters, not TeX groups.
                    groups = re.sub(r'\\[{}]', '', block)
                    for char in groups:
                        depth += (char == '{') - (char == '}')
                        self.assertGreaterEqual(depth, 0)
                    self.assertEqual(depth, 0)
                    self.assertEqual(
                        len(re.findall(r'\\left(?![A-Za-z])', block)),
                        len(re.findall(r'\\right(?![A-Za-z])', block)),
                    )
                    self.assertEqual(
                        re.findall(r'\\begin\{([^}]+)\}', block),
                        list(reversed(re.findall(r'\\end\{([^}]+)\}', block))),
                    )
                total += len(blocks)
        self.assertEqual(total, 19)


if __name__ == '__main__':
    unittest.main()
