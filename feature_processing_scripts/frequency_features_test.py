import unittest
from feature_processing_scripts.frequency_features import get_segments

class testing_frequency_features(unittest.TestCase):

    def test_upper(self):
        data = range(0, 103)
        segments = get_segments(data)
        self.assertEqual(first=[0, 1, 2, 3, 4], second=segments[100])
        self.assertEqual(first=[15, 16, 17, 18, 19], second=segments[85])
        self.assertEqual(first=[35, 36, 37, 38, 39], second=segments[65])
        self.assertEqual(first=[95,96,97,98,99,100,101,102], second=segments[5])

if __name__ == '__main__':
    unittest.main()
