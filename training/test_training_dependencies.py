"""Import smoke tests for optional packages used on active training paths."""
import unittest


class TrainingDependencyTests(unittest.TestCase):
    def test_json_schema_sampling_dependency(self):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        self.assertTrue(callable(JsonSchemaParser))
        self.assertTrue(callable(build_transformers_prefix_allowed_tokens_fn))


if __name__ == "__main__":
    unittest.main()
