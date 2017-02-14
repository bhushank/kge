__author__ = 'bhushan'

import unittest
from test.models import TestModels



if __name__=='__main__':

    runner = unittest.TextTestRunner(verbosity=3)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestModels))

    runner.run(suite)