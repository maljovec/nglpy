""" This module will test the functionality of nglpy.utils
"""
import unittest
import warnings

import nglpy


class TestUtils(unittest.TestCase):
    """Class for testing the utils"""

    def test_consume_extra_args_with_exception(self):
        """Tests the consume_extra_args function raises an exception."""
        unused = {"extra": "word"}
        msg = "Warning: the current version of nglpy does not accept"
        with self.assertRaises(NotImplementedError) as context:
            nglpy.utils.consume_extra_args(fail_on_missing=True, **unused)
            self.assertTrue(msg in context.exception)

    def test_consume_extra_args(self):
        """Tests the consume_extra_args function raises a warning."""
        unused = {"extra": "word"}
        msg = "Warning: the current version of nglpy does not accept"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nglpy.utils.consume_extra_args(fail_on_missing=False, **unused)

            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, UserWarning))
            self.assertTrue(msg in str(w[-1].message))


if __name__ == "__main__":
    unittest.main()
