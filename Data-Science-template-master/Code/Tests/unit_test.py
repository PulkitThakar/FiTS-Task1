 ### Main class of unit testing, which calls the different unit testing function of test_function


import unittest
from test_functions import *
import warnings

class Test_Model(unittest.TestCase):

    def test_deployment(self):
        self.assertTrue(checkdeployment())

    def test_TrainingDataFormat(self):
        self.assertTrue(checkTrainingDataFormat())

    def test_ModelSaving(self):
        self.assertTrue(checkModelSaving())

    def test_Precision(self):
        self.assertTrue(checkPrecision())
    
    def test_TrainingMethod(self):
        self.assertTrue(checkTrainingMethod())
  
  
#the following is not required if call by pytest instead of python
if __name__ == '__main__':
    unittest.main()
