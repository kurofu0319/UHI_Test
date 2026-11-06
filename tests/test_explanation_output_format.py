import unittest

from scripts.infer_explanation_samples import (
    EXPLANATION_GUIDELINES,
    PREDICTION_SUFFIX,
    SECTION_HEADINGS,
    add_explanation_guidelines,
    validate_explanation_output,
)
from src.output_parsing import parse_prediction


def valid_output():
    sections = "\n\n".join(
        f"{index}. {heading}\nA non-empty explanation for section {index}."
        for index, heading in enumerate(SECTION_HEADINGS, start=1)
    )
    return f"-0.125 °C\n\n{sections}"


class ExplanationOutputFormatTest(unittest.TestCase):
    def test_guidelines_are_inserted_before_prediction_cue(self):
        prompt = "Task text" + PREDICTION_SUFFIX

        guided = add_explanation_guidelines(prompt)

        self.assertTrue(guided.endswith(PREDICTION_SUFFIX))
        self.assertIn(EXPLANATION_GUIDELINES, guided)
        self.assertLess(guided.index(EXPLANATION_GUIDELINES), guided.index(PREDICTION_SUFFIX))

    def test_accepts_numbered_six_section_response(self):
        output = valid_output()
        parsed = parse_prediction(output)

        validation = validate_explanation_output(output, parsed)

        self.assertTrue(validation["format_valid"])
        self.assertTrue(validation["headings_in_order"])
        self.assertTrue(validation["sections_nonempty"])
        self.assertEqual(validation["format_errors"], [])

    def test_accepts_period_after_celsius_unit(self):
        output = valid_output().replace("°C\n", "°C.\n", 1)
        parsed = parse_prediction(output)

        validation = validate_explanation_output(output, parsed)

        self.assertTrue(validation["format_valid"])

    def test_accepts_bold_markdown_section_headings(self):
        output = valid_output()
        for index, heading in enumerate(SECTION_HEADINGS, start=1):
            output = output.replace(
                f"{index}. {heading}",
                f"{index}. **{heading}**",
            )
        parsed = parse_prediction(output)

        validation = validate_explanation_output(output, parsed)

        self.assertTrue(validation["format_valid"])

    def test_rejects_missing_unit_duplicate_heading_and_empty_section(self):
        output = valid_output().replace("-0.125 °C", "-0.125", 1)
        output = output.replace(
            "1. Geographical Context:\nA non-empty explanation for section 1.",
            "1. Geographical Context:\n\n1. Geographical Context:\nRepeated.",
            1,
        )
        parsed = parse_prediction(output)

        validation = validate_explanation_output(output, parsed)

        self.assertFalse(validation["format_valid"])
        self.assertIn("missing_result_value_or_celsius_unit", validation["format_errors"])
        self.assertIn(
            "heading_count:Geographical Context::2",
            validation["format_errors"],
        )

    def test_rejects_more_than_three_decimal_places(self):
        output = valid_output().replace("-0.125", "-0.1259", 1)
        parsed = parse_prediction(output)

        validation = validate_explanation_output(output, parsed)

        self.assertFalse(validation["format_valid"])
        self.assertIn(
            "prediction_exceeds_three_decimal_places",
            validation["format_errors"],
        )


if __name__ == "__main__":
    unittest.main()
