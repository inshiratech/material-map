# Material Map

A Streamlit MVP to help manufacturers quantify scrap, visualize material flow, and surface ROI opportunities from their production data.

## Features
- Guided uploads with schema validation for material, process steps, QC, and output data (CSV or Excel) plus downloadable templates.
- Dynamic Sankey visualization that derives transitions from your process steps and colors links by loss rates.
- KPI cards with benchmarking, adjustable scrap-reduction targets, and ROI estimates based on your material costs and annual volume.
- Batch-level table with highlights for the worst performers and a downloadable report that includes KPIs and batch details.
- In-app feedback capture so pilot users can share what’s missing.

## Getting started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in your browser.

## Data expectations
Each upload must include the following columns (extra columns are ignored):

| File | Required columns |
| --- | --- |
| Material Data | `Batch ID`, `Initial Quantity` |
| Process Steps | `Batch ID`, `Process Step` (optional: `Step Order` for sequencing) |
| QC Reports | `Batch ID`, `Scrap Quantity` |
| Final Output | `Batch ID`, `Final Quantity` |

Template CSVs are available in the app to align exports quickly. If no process file is supplied, the app falls back to a default Delivery → Cutting → Milling → QC → Finished Product path and flags the user.

## Usage tips
- Enter your material cost per unit and annual batch volume to translate waste into currency.
- Use the scrap-reduction slider in the Benchmarks & ROI section to see target-driven savings.
- Download the combined KPI + batch-level CSV report for offline sharing.
- Submit feedback through the built-in form; responses are saved to `feedback_responses.csv` in the project root.

## Deployment
A `Procfile` is included for quick Heroku/Streamlit Cloud deployment:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
Set any required secrets (e.g., telemetry tokens) as environment variables in your hosting platform.
