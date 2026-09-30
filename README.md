# 🏨 Hotel Booking Cancellation Predictor

A Streamlit app built from the `Finpro_Hotel_Booking_Cancellation` notebook. It predicts which hotel
bookings are likely to be cancelled **at the time of booking**, so the hotel can focus retention efforts
and reduce empty rooms.

## App pages

| Page | What it shows |
| --- | --- |
| **Overview** | Business problem, key numbers, data-cleaning steps, recommendations |
| **EDA** | The 5 business questions (lead time, hotel type, season, deposit, market segment) |
| **Model** | Model benchmark, cost-based threshold calculator (adjustable costs), feature importance |
| **Predict** | Score a single booking with a form, or upload a CSV to score many bookings |

## Project structure

```
├── app.py                 # Streamlit app
├── utils.py               # Data cleaning + prediction helpers (shared with train_model.py)
├── train_model.py         # Rebuilds the final model from the dataset
├── requirements.txt       # Pinned versions (must match the .pkl file)
├── data/hotel_bookings_dataset.csv
└── model/final_model_hotel_cancellation.pkl
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free)

1. Create a **public or private GitHub repo** and push this whole folder (the CSV is ~17 MB and the model ~3 MB, both fine for GitHub).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Choose your repo, branch `main`, and main file path `app.py`.
5. Open **Advanced settings** and select **Python 3.12**.
6. Click **Deploy**. The first build takes a few minutes; afterwards every `git push` redeploys automatically.

## Retrain the model

```bash
python train_model.py
```

This recreates `model/final_model_hotel_cancellation.pkl` using the notebook's cleaning steps, train/test split
(`random_state=42`) and the best LightGBM parameters from Section 10.2. If you change any package version in
`requirements.txt`, retrain so the `.pkl` matches the installed scikit-learn / LightGBM versions.

## Notes

- The risk score comes from a class-weighted model, so it is used for **ranking**, not as a calibrated probability.
  The Predict page therefore also shows the actual cancellation rate of test-set bookings with a similar score.
- The decision threshold (0.20) comes from the notebook's assumed costs (retention effort = 1, empty room = 10).
  Replace these with the hotel's real numbers on the Model page to see how the optimal threshold changes.
