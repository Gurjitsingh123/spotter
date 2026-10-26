import json
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
results = json.loads((ROOT / "reports" / "metrics.json").read_text())
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class Doc(FPDF):
    def heading(self, text, size=13):
        self.ln(3); self.set_font("B", size=size); self.set_text_color(6, 74, 86); self.multi_cell(0, 7, text, new_x="LMARGIN", new_y="NEXT"); self.set_text_color(20, 20, 20)

    def para(self, text):
        self.set_font("results", size=10); self.multi_cell(0, 5.2, text, new_x="LMARGIN", new_y="NEXT"); self.ln(1.5)

    def table(self, header, rows, widths):
        self.set_font("B", size=9); self.set_fill_color(230, 238, 240)
        for h, w in zip(header, widths):
            self.cell(w, 7, h, border=1, fill=True)
        self.ln()
        self.set_font("results", size=9)
        for row in rows:
            for v, w in zip(row, widths):
                self.cell(w, 6.5, str(v), border=1)
            self.ln()
        self.ln(2)


data_summary = results["data"]
pdf = Doc(format="A4")
pdf.set_margins(18, 16, 18)
pdf.add_font("results", "", FONT)
pdf.add_font("B", "", FONT_B)
pdf.add_page()
pdf.set_font("B", size=18); pdf.set_text_color(6, 74, 86); pdf.cell(0, 10, "Freight Rate Prediction — Assessment Report", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("results", size=10); pdf.set_text_color(90, 90, 90); pdf.cell(0, 6, "Machine Learning Engineer assessment · Spotter · Gurjit Singh", new_x="LMARGIN", new_y="NEXT"); pdf.set_text_color(20, 20, 20)

pdf.heading("1. Summary")
unseen = results["unseen_city_split"]; folds = results["rolling_time_split"]
latest_month = folds[-1]
pdf.para(f"A LightGBM model predicts each load's rate per mile, multiplied back by distance. On a strict time-based split it reaches "
      f"{latest_month['MAPE_clean_rows_pct']}% MAPE (MAE ${latest_month['MAE_clean_rows']:.0f}) on the latest month (Oct 2025) and "
      f"{unseen['MAPE_clean_rows_pct']}% MAPE on loads involving cities it never saw in training. Removing ~1.3% corrupt labels was the single "
      f"biggest improvement (holdout MAPE 4.2% → 2.3%).")

pdf.heading("2. Data and key findings")
pdf.para(f"Training data: {data_summary['train_rows']:,} labelled loads from {data_summary['train_dates'][0]} to {data_summary['train_dates'][1]}. Validation: "
      f"{data_summary['validation_rows']:,} loads from {data_summary['validation_dates'][0]} to {data_summary['validation_dates'][1]} — entirely in the future relative to training. "
      f"Distance explains most of the price (correlation 0.91 with rate). Reefer is the most expensive equipment per mile, then Flatbed, then Dry Van. "
      f"market_index is essentially one value per day and tracks the daily median price per mile (correlation ≈ 0.7), with a weekly cycle. "
      f"Eight cities in the validation set never appear in training, so the model must generalise beyond city names.")
pdf.image(str(ROOT / "reports" / "rpm_vs_market_index.png"), w=170)

pdf.heading("3. Data-quality issues and how they were handled")
pdf.table(["Issue", "Evidence", "Handling"], [
    ["Corrupt rate labels", f"{data_summary['label_outliers_removed']} rows: $/mile 2–7x off", "Removed from training only"],
    ["Negative weights", f"{data_summary['negative_weights_fixed']} rows (down to -47,500 lb)", "Absolute value (sign error)"],
    ["Missing weight", f"{data_summary['missing_weight']} rows", "Median weight for that equipment"],
    ["Missing market_index", f"{data_summary['missing_market_index']} rows", "That day's mean (index is per-day)"],
    ["Coordinates vs distance", "Short lanes: coords drift", "Trust given road distance"],
    ["Unseen cities", "8 cities only in validation", "Coordinate features, no city IDs"],
], [44, 64, 66])
pdf.image(str(ROOT / "reports" / "rate_per_mile_distribution.png"), w=150)
pdf.para("One cleaning idea was tested and rejected: rebuilding distances from coordinates where they disagreed. Those rows were all short "
      "lanes (~70–90 miles) whose road distance is consistent across loads on the same lane — the synthetic coordinates were the noisy "
      "part — so the 'fix' made predictions worse and was removed.")

pdf.add_page()
pdf.heading("4. Train / validation split")
pdf.para("Because the final validation set (Nov–Dec 2025) lies after all training data (Jan–Oct 2025), a random split would leak future market "
      "conditions and overstate accuracy. I used an expanding-window time split: for each test month M (Jul, Aug, Sep, Oct), the model is "
      "trained on every earlier month and evaluated on M. Errors are reported on clean rows (outlier labels excluded, since no model can "
      "predict a data-entry error) and on all rows. A second test removes 8 random cities from training and evaluates Sep–Oct loads that "
      "touch them, mirroring the unseen cities in the validation file. Label-outlier rules are fitted on the training part of each split only.")
pdf.table(["Test month", "MAPE clean", "MAE clean", "MAE all rows", "Rows"],
          [[fold["test_month"], f"{fold['MAPE_clean_rows_pct']}%", f"${fold['MAE_clean_rows']:.0f}", f"${fold['MAE_all_rows']:.0f}", fold["rows"]] for fold in folds]
          + [["Unseen cities", f"{unseen['MAPE_clean_rows_pct']}%", f"${unseen['MAE_clean_rows']:.0f}", f"${unseen['MAE_all_rows']:.0f}", unseen["rows"]]],
          [34, 30, 30, 34, 24])
pdf.para("August is the weak month (5.7% MAPE): the market index fell sharply from July and prices adjusted differently than earlier in the year, "
      "a regime change that models trained only on Jan–Jul could not anticipate. Adding 7- and 28-day rolling market features was tested; it "
      "did not fix August and degraded later months (Oct 1.97% → 4.67%), so it was not used.")

pdf.heading("5. Model choice")
pdf.table(["Model (train Jan–Aug, test Sep–Oct)", "MAPE clean", "MAE clean"], [
    ["Median $/mile by equipment x distance", "8.15%", "$176"],
    ["Ridge regression on log(rate)", "4.54%", "$106"],
    ["LightGBM on log(rate)", "2.60%", "$59"],
    ["LightGBM on rate per mile (chosen)", "2.31%", "$54"],
    ["Same, but corrupt labels kept", "4.23%", "$97"],
], [90, 40, 40])
pdf.para("Gradient-boosted trees capture the non-linear interplay of distance, equipment, region and market level without manual interactions. "
      "Predicting rate per mile makes the target comparable across short and long hauls, so the trees spend their capacity on the price level "
      "rather than on re-learning distance. Top features by gain: " +
      ", ".join(f"{k} ({v:.0f}%)" for k, v in list(results["feature_importance_gain_pct"].items())[:6]) + ".")

pdf.heading("6. December 2025 prediction chart (from score.py)")
pdf.para("Fixed load: Lexington → Fort Wayne, 360 miles, Dry Van, 32,000 lb; only the date changes. The December inputs have no coordinates or "
      "market signals, so city coordinates come from the training data and each day's market_index / quote_signal mean comes from the "
      "validation file, which already covers December. Predictions range "
      f"${min(x['predicted_rate'] for x in results['december_predictions']):.0f}–${max(x['predicted_rate'] for x in results['december_predictions']):.0f} "
      "(≈ $2.15–2.26 per mile) with the weekly market rhythm; historical Dry Van loads on this lane averaged $808.")
pdf.image(str(ROOT / "scorer_results" / "candidate_december.png"), w=174)

pdf.heading("7. Limitations and next steps")
pdf.para("• Regime shifts (like August) remain the main error source — external market data (diesel prices, load-to-truck ratios) would help.\n"
      "• Lane-level history could be added as a smoothed target encoding with time-aware cross-fitting.\n"
      "• Prediction intervals (quantile LightGBM) would let pricing teams see uncertainty, not just a point estimate.")
out = ROOT / "reports" / "Freight_Rate_Report.pdf"
pdf.output(str(out))
print("wrote", out)
