# Freight Document Extraction - Eval Harness

A prototype that evaluates prompt strategies for extracting structured fields from freight documents: carrier invoices, BOLs, and rate confirmations.

Built to explore where VLM-based extraction breaks on real-world freight documents, and whether better prompting fixes it or whether the problem is structural.

---

## What it does

Runs freight document images through 3 prompt strategies using Gemini Vision. Scores each field against known ground truth and categorises each failure type.

**Fields extracted per document**

| Field | Notes |
|---|---|
| Carrier name | Legal name of carrier |
| Linehaul rate | Base transportation charge |
| Fuel surcharge | May be labeled FSC, Fuel Adj, Fuel Surcharge |
| Accessorial charges | Sum of all extra charges |
| Weight | Shipment weight in lbs |
| Total amount | Final amount due |

**Prompt strategies compared**

| Strategy | Approach |
|---|---|
| A - Naive | Simple field list, minimal instruction |
| B - Structured | Typed schema with explicit handling rules |
| C - Few-shot | Example-guided, precise output patterns |

**Failure taxonomy**

- `layout_ambiguity`: field spans multiple columns, not a prompt problem
- `format_variance`: same field labeled differently across carriers
- `scan_quality`: rotated, low-res, or occluded documents
- `prompt_fixable`: Strategy C succeeds where A and B fail

---

## Project structure

```
app.py              Streamlit app - batch eval + live upload
prompts.py          3 prompt strategies
ground_truth.json   Known field values for your test documents
requirements.txt
.env.example        API key template
```

---

*Prototype. Built to understand the eval problem in freight document intelligence.*  
*Not production-ready.*
