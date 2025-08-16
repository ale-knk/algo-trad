import unittest

from pytrad.indicator_name import IndicatorName


class TestIndicatorName(unittest.TestCase):
    def test_basic_init(self):
        """Test basic initialization of IndicatorName."""
        ind_name = IndicatorName("RSI", [14], False)
        self.assertEqual(ind_name.base_name, "RSI")
        self.assertEqual(ind_name.parameters, [14])
        self.assertFalse(ind_name.uses_returns)
        self.assertIsNone(ind_name.component)

    def test_to_string_simple(self):
        """Test string conversion for simple indicators."""
        # Simple indicator
        ind_name = IndicatorName("MA", [20])
        self.assertEqual(ind_name.to_string(), "MA_20")

        # With returns
        ind_name = IndicatorName("RSI", [14], True)
        self.assertEqual(ind_name.to_string(), "RSI_14-returns")

    def test_to_string_complex(self):
        """Test string conversion for complex indicators with components."""
        # Bollinger Bands with component
        ind_name = IndicatorName("BB", [20, 2], False, "upper_band")
        self.assertEqual(ind_name.to_string(), "BB_20_2-upper_band")

        # MACD with component and returns
        ind_name = IndicatorName("MACD", [12, 26, 9], True, "signal_line")
        self.assertEqual(ind_name.to_string(), "MACD_12_26_9-returns-signal_line")

    def test_from_string_simple(self):
        """Test parsing simple indicator strings."""
        # Simple indicator
        ind_name = IndicatorName.from_string("MA_20")
        self.assertEqual(ind_name.base_name, "MA")
        self.assertEqual(ind_name.parameters, [20])
        self.assertFalse(ind_name.uses_returns)
        self.assertIsNone(ind_name.component)

        # With returns
        ind_name = IndicatorName.from_string("RSI_14-returns")
        self.assertEqual(ind_name.base_name, "RSI")
        self.assertEqual(ind_name.parameters, [14])
        self.assertTrue(ind_name.uses_returns)
        self.assertIsNone(ind_name.component)

    def test_from_string_complex(self):
        """Test parsing complex indicator strings with components."""
        # Bollinger Bands with component
        ind_name = IndicatorName.from_string("BB_20_2-upper_band")
        self.assertEqual(ind_name.base_name, "BB")
        self.assertEqual(ind_name.parameters, [20, 2])
        self.assertFalse(ind_name.uses_returns)
        self.assertEqual(ind_name.component, "upper_band")

        # MACD with component and returns
        ind_name = IndicatorName.from_string("MACD_12_26_9-returns-signal_line")
        self.assertEqual(ind_name.base_name, "MACD")
        self.assertEqual(ind_name.parameters, [12, 26, 9])
        self.assertTrue(ind_name.uses_returns)
        self.assertEqual(ind_name.component, "signal_line")

    def test_equality(self):
        """Test equality comparison between IndicatorName objects."""
        ind1 = IndicatorName("RSI", [14])
        ind2 = IndicatorName("RSI", [14])
        ind3 = IndicatorName("RSI", [21])
        ind4 = IndicatorName("RSI", [14], True)

        self.assertEqual(ind1, ind2)
        self.assertNotEqual(ind1, ind3)
        self.assertNotEqual(ind1, ind4)

        # Compare with string
        self.assertEqual(ind1, "RSI_14")
        self.assertNotEqual(ind1, "RSI_21")
        self.assertNotEqual(ind1, "RSI_14-returns")

    def test_hash(self):
        """Test that IndicatorName objects can be used as dictionary keys."""
        ind1 = IndicatorName("RSI", [14])
        ind2 = IndicatorName("RSI", [14])
        ind3 = IndicatorName("RSI", [21])

        # Create a dictionary with IndicatorName keys
        indicator_values = {ind1: [1, 2, 3], ind3: [4, 5, 6]}

        # Test lookup with equal object
        self.assertEqual(indicator_values[ind2], [1, 2, 3])
        self.assertEqual(len(indicator_values), 2)

    def test_with_component(self):
        """Test creating a new IndicatorName with a component."""
        ind = IndicatorName("BB", [20, 2])
        ind_upper = ind.with_component("upper_band")

        self.assertEqual(ind_upper.base_name, "BB")
        self.assertEqual(ind_upper.parameters, [20, 2])
        self.assertFalse(ind_upper.uses_returns)
        self.assertEqual(ind_upper.component, "upper_band")

    def test_without_component(self):
        """Test creating a new IndicatorName without a component."""
        ind = IndicatorName("BB", [20, 2], False, "upper_band")
        ind_base = ind.without_component()

        self.assertEqual(ind_base.base_name, "BB")
        self.assertEqual(ind_base.parameters, [20, 2])
        self.assertFalse(ind_base.uses_returns)
        self.assertIsNone(ind_base.component)

    def test_is_complex(self):
        """Test the is_complex property."""
        ind_simple = IndicatorName("RSI", [14])
        ind_complex = IndicatorName("BB", [20, 2], False, "upper_band")

        self.assertFalse(ind_simple.is_complex)
        self.assertTrue(ind_complex.is_complex)


if __name__ == "__main__":
    unittest.main()
