import unittest

class TestSimple(unittest.TestCase):
    def testImport(self) -> None:
        import anitune  # noqa


if __name__ == "__main__":
    unittest.main()
