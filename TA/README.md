# Monero Technical Analysis Snapshot Generator

This project generates a technical analysis (TA) snapshot for monero, including a comprehensive PNG chart and a Markdown summary. It fetches price data, computes key indicators and outputs assets for newsletter publication.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python ta_script.py
   ```
3. Find the generated chart and Markdown file in the dated folder under `newsletter_assets/ta/`

## Output
- `xmr_comprehensive.png`: Chart with price and indicators
- `xmr_ta_YYYY-MM-DD.md`: Markdown summary for newsletter

## Requirements
- Python 3.7+
- See `requirements.txt` for required packages

## Folder Structure
```
TA/
  requirements.txt
  ta_script.py
  newsletter_assets/
    ta/
      YYYY-MM-DD/
        manifest.json
        xmr_comprehensive.png
        xmr_ta_YYYY-MM-DD.md
```

## License
MIT

## Author
CypherGoat
