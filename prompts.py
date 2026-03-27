# Three prompt strategies for freight document field extraction.
# Each targets the same 6 fields with increasing specificity.

FIELDS = [
    "carrier_name",
    "linehaul_rate",
    "fuel_surcharge",
    "accessorial_charges",
    "weight",
    "total_amount",
]

FIELD_LABELS = {
    "carrier_name":        "Carrier name",
    "linehaul_rate":       "Linehaul rate",
    "fuel_surcharge":      "Fuel surcharge",
    "accessorial_charges": "Accessorial charges",
    "weight":              "Weight",
    "total_amount":        "Total amount",
}

# Strategy A — Naive
# Simple field list, minimal instruction.
STRATEGY_A = """Extract the following fields from this freight document.
Return as a JSON object only. No explanation, no markdown.

Fields:
- carrier_name: name of the carrier or trucking company
- linehaul_rate: base transportation charge (number only, no $ sign)
- fuel_surcharge: fuel surcharge amount (number only, no $ sign)
- accessorial_charges: total of all extra charges like liftgate, detention, residential (number only)
- weight: shipment weight in pounds (number only)
- total_amount: final total amount due (number only)

Use null for any field not found."""


# Strategy B — Structured
# Typed schema with field descriptions and explicit handling rules.
STRATEGY_B = """You are a freight document parser. Extract structured data from this image.

Return ONLY a valid JSON object with these exact keys. No explanation, no markdown.

{
  "carrier_name": "Legal name of the carrier company (string)",
  "linehaul_rate": "Base transportation charge in USD, numbers only (float or null)",
  "fuel_surcharge": "Fuel surcharge — may be labeled FSC, Fuel Adj, or Fuel Surcharge (float or null)",
  "accessorial_charges": "Sum of ALL extra line items: liftgate, detention, residential delivery, limited access, TONU, layover, stop-off fee (float or null)",
  "weight": "Total shipment weight in pounds (float or null)",
  "total_amount": "Final amount due — look for Total, Amount Due, Balance (float or null)"
}

Rules:
- For accessorial_charges: add up every charge that is not linehaul or fuel
- Strip all $ signs and units from numeric fields
- Return null, not 0, when a field is genuinely absent"""


# Strategy C — Few-shot
# Example-guided extraction. Gives the model precise output patterns to follow.
STRATEGY_C = """Extract freight document fields and return as JSON only. No explanation, no markdown.

--- Example 1 ---
Document: ABC Trucking LLC, linehaul $1,200.00, FSC $180.00, liftgate $75.00, 24,000 lbs, total $1,455.00
Output: {"carrier_name": "ABC Trucking LLC", "linehaul_rate": 1200.0, "fuel_surcharge": 180.0, "accessorial_charges": 75.0, "weight": 24000.0, "total_amount": 1455.0}

--- Example 2 ---
Document: XYZ Freight, flat rate $850, fuel included, no extras, 18,500 lbs
Output: {"carrier_name": "XYZ Freight", "linehaul_rate": 850.0, "fuel_surcharge": 0.0, "accessorial_charges": 0.0, "weight": 18500.0, "total_amount": 850.0}

--- Example 3 ---
Document: Fast Haul Inc, linehaul $1,100, fuel surcharge $160, detention $75, residential $50, 31,000 lbs, total $1,385
Output: {"carrier_name": "Fast Haul Inc", "linehaul_rate": 1100.0, "fuel_surcharge": 160.0, "accessorial_charges": 125.0, "weight": 31000.0, "total_amount": 1385.0}

--- Example 4 ---
Document: Midwest Carriers, weight 42,500 lbs, no pricing shown
Output: {"carrier_name": "Midwest Carriers", "linehaul_rate": null, "fuel_surcharge": null, "accessorial_charges": null, "weight": 42500.0, "total_amount": null}

Now extract from the provided document. Return ONLY the JSON object."""


STRATEGIES = {
    "A — Naive":      STRATEGY_A,
    "B — Structured": STRATEGY_B,
    "C — Few-shot":   STRATEGY_C,
}
