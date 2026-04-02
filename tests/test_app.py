import unittest

class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        # Code to set up the Flask app instance
        pass

    def test_home_page(self):
        # Test the home page of the Flask app
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_not_found_page(self):
        # Test a non-existent page
        response = self.app.get('/non-existent')
        self.assertEqual(response.status_code, 404)

if __name__ == '__main__':
    unittest.main()