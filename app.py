import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from utils import MONTHS, cost_of_predictions, load_and_clean, predict_risk, threshold_table

MODEL_PATH = "model/final_model_hotel_cancellation.pkl"
BLUE = "#4C78A8"

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")


# ---------- Loading (cached so it only runs once) ----------
@st.cache_data
def get_data():
    return load_and_clean()


@st.cache_resource
def get_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run `python train_model.py` first.")
    st.stop()

df, cleaning_steps = get_data()
artifact = get_artifact()


# ---------- Small chart helper ----------
def bar_chart(values, title, ylabel="Cancellation rate (%)", horizontal=False, fmt="%.1f"):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = [str(i) for i in values.index]
    if horizontal:
        bars = ax.barh(labels, values.values, color=BLUE)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
    else:
        bars = ax.bar(labels, values.values, color=BLUE)
        ax.set_ylabel(ylabel)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.bar_label(bars, fmt=fmt, padding=2, fontsize=8)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def cancel_rate(column):
    """Cancellation rate (%) per category of a column."""
    return df.groupby(column)["is_canceled"].mean() * 100


# =====================================================================
# PAGE 1 - OVERVIEW
# =====================================================================
def page_overview():
    st.title("🏨 Hotel Booking Cancellation")
    st.write(
        "Cancelled bookings leave hotel rooms empty, and an unsold room can never be sold again. "
        "This app predicts **which bookings are likely to be cancelled at the time they are made**, "
        "so the hotel can focus retention efforts (reconfirmation, deposits, controlled overbooking) "
        "where they matter most."
    )

    raw_rate = cleaning_steps[0]["Cancellation rate"]
    clean_rate = cleaning_steps[-1]["Cancellation rate"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bookings (after cleaning)", f"{len(df):,}")
    c2.metric("Cancellation rate (raw)", f"{raw_rate:.1%}")
    c3.metric("Cancellation rate (cleaned)", f"{clean_rate:.1%}")
    c4.metric("Final model", artifact["model_name"])

    st.subheader("Data cleaning")
    steps = pd.DataFrame(cleaning_steps)
    steps["Rows"] = steps["Rows"].map("{:,}".format)
    steps["Cancellation rate"] = (steps["Cancellation rate"] * 100).round(1).astype(str) + "%"
    st.dataframe(steps, hide_index=True)
    st.caption(
        "Almost the whole drop from 37% to 27.6% comes from removing duplicate rows. "
        "Leakage columns (e.g. `reservation_status`) are also dropped, because they are only known after the outcome."
    )

    st.subheader("Business recommendations")
    st.markdown(
        """
1. **Reconfirm long lead-time bookings** (WhatsApp / phone / email): the longer the lead time, the higher the risk.
2. **Prioritise Online TA and Groups**: highest cancellation rate combined with high volume.
3. **Audit Non Refund deposits**: a ~95% cancellation rate is abnormal and concentrated in one country and channel.
4. **Controlled overbooking in peak season (Apr–Aug)**: use the historical cancellation rate per risk-score band to estimate expected cancellations.
5. **Score every new booking** on the *Predict* page and route risky ones to a retention flow.
        """
    )


# =====================================================================
# PAGE 2 - EDA (5 business questions)
# =====================================================================
def page_eda():
    st.title("📊 Exploratory Analysis")
    st.write("Five business questions about what drives cancellations (cleaned data).")

    q1, q2, q3, q4, q5 = st.tabs(
        ["1. Lead time", "2. Hotel type", "3. Season", "4. Deposit", "5. Market segment"]
    )

    with q1:
        st.subheader("Does booking far in advance increase cancellation risk?")
        bins = [-1, 7, 30, 90, 180, 365, 10000]
        labels = ["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", ">365 days"]
        lead_bin = pd.cut(df["lead_time"], bins=bins, labels=labels)
        rate = df.groupby(lead_bin, observed=True)["is_canceled"].mean() * 100
        bar_chart(rate, "Cancellation rate by lead time")
        st.info(
            "Yes, and almost monotonically: from about 8% for last-minute bookings to about 40% "
            "for bookings made more than 6 months ahead. Bookings above 90 days deserve extra "
            "mitigation (deposit, reconfirmation at D-30 / D-7)."
        )

    with q2:
        st.subheader("Do City and Resort hotels behave differently?")
        bar_chart(cancel_rate("hotel"), "Cancellation rate by hotel type")
        st.info(
            "City Hotel (30.1%) is more cancellation-prone than Resort Hotel (23.7%). "
            "The exception is the `Group` customer type, where Resort is slightly higher."
        )

    with q3:
        st.subheader("Which months have the highest cancellation rate?")
        rate = cancel_rate("arrival_date_month").reindex(MONTHS)
        bar_chart(rate, "Cancellation rate by arrival month")
        st.info(
            "April–August (peak season) sit around 29–32%, versus roughly 21–25% in the low months. "
            "The pattern is consistent but not extreme."
        )

    with q4:
        st.subheader("Do deposits reduce cancellations?")
        bar_chart(cancel_rate("deposit_type"), "Cancellation rate by deposit type")
        counts = df["deposit_type"].value_counts().rename("Bookings").to_frame()
        st.dataframe(counts)
        st.info(
            "Surprisingly, **Non Refund** bookings are cancelled ~95% of the time. This is very unlikely to be "
            "guest psychology: those bookings are concentrated in Portugal via travel agents / tour operators, "
            "which points to a specific agent or channel practice worth auditing."
        )

    with q5:
        st.subheader("Which market segments contribute most to cancellations?")
        rate = cancel_rate("market_segment").sort_values(ascending=False)
        bar_chart(rate, "Cancellation rate by market segment", ylabel="Cancellation rate (%)", horizontal=True)
        counts = df["market_segment"].value_counts().rename("Bookings").to_frame()
        st.dataframe(counts)
        st.info(
            "**Online TA** has the highest rate (35%) *and* by far the largest volume, so it is the top mitigation "
            "priority. **Groups** come second; Direct, Corporate and Offline TA/TO are low."
        )

    with st.expander("Preview cleaned data"):
        st.dataframe(df.head(100))


# =====================================================================
# PAGE 3 - MODEL
# =====================================================================
BENCHMARK = pd.DataFrame(
    [
        ["CatBoost", 0.838, 0.605, 0.703, 0.898, 0.779],
        ["XGBoost", 0.826, 0.601, 0.696, 0.895, 0.771],
        ["LightGBM", 0.832, 0.591, 0.691, 0.891, 0.763],
        ["Random Forest", 0.838, 0.541, 0.658, 0.878, 0.737],
        ["Gradient Boosting", 0.542, 0.732, 0.623, 0.874, 0.734],
        ["Decision Tree", 0.785, 0.570, 0.660, 0.860, 0.684],
        ["KNN", 0.540, 0.697, 0.608, 0.845, 0.679],
        ["Logistic Regression", 0.776, 0.528, 0.628, 0.837, 0.665],
    ],
    columns=["Model", "Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"],
)

TUNED_CV = pd.DataFrame(
    [
        ["LightGBM (tuned)", 0.768, 0.894, 0.696],
        ["XGBoost (tuned)", 0.766, 0.893, 0.694],
        ["CatBoost (tuned)", 0.750, 0.885, 0.682],
    ],
    columns=["Model", "PR-AUC", "ROC-AUC", "F1"],
)


def page_model():
    st.title("🤖 Model")
    y_test = artifact["y_test"]
    y_prob = artifact["y_prob_test"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Final model", f"{artifact['model_name']} (tuned)")
    c2.metric("ROC-AUC (test set)", f"{roc_auc_score(y_test, y_prob):.3f}")
    c3.metric("PR-AUC (test set)", f"{average_precision_score(y_test, y_prob):.3f}")

    st.subheader("How the model was chosen")
    st.write("Eight models were benchmarked (test set, threshold 0.5, results taken from the notebook):")
    st.dataframe(BENCHMARK, hide_index=True)
    st.write(
        "The top three (CatBoost, XGBoost, LightGBM) were tuned and cross-validated. "
        "After tuning, **LightGBM** had the best 5-fold CV scores, so it became the final model:"
    )
    st.dataframe(TUNED_CV, hide_index=True)

    # ----- Cost-based threshold -----
    st.subheader("Choosing the decision threshold from business cost")
    st.write(
        "A missed cancellation (empty room) costs much more than a wasted retention effort, "
        "so the best threshold is well below 0.5. Change the costs to see how the optimum moves."
    )
    col1, col2 = st.columns(2)
    retention_cost = col1.number_input("Cost of a retention effort (per flagged guest)", 0.1, 1000.0, 1.0, 0.5)
    empty_room_cost = col2.number_input("Cost of an unexpected empty room", 0.1, 10000.0, 10.0, 1.0)

    table = threshold_table(y_test, y_prob, retention_cost, empty_room_cost)
    best = table.loc[table["Net benefit"].idxmax()]
    best_t = float(best["Threshold"])

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(table["Threshold"], table["Net benefit"], marker="o", color=BLUE)
    ax.axvline(best_t, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Net benefit vs threshold (best = {best_t})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    y_pred = (y_prob >= best_t).astype(int)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best threshold", f"{best_t}")
    m2.metric("Recall (cancellations caught)", f"{recall_score(y_test, y_pred):.1%}")
    m3.metric("Precision", f"{precision_score(y_test, y_pred):.1%}")
    m4.metric("Missed cancellations (FN)", f"{int(best['FN']):,}")

    st.write("**Model vs. naive strategies (same test set):**")
    n = len(y_test)
    comparison = pd.DataFrame(
        [
            {"Strategy": "Assume nobody cancels", **cost_of_predictions(y_test, np.zeros(n, dtype=int), retention_cost, empty_room_cost)},
            {"Strategy": "Assume everybody cancels", **cost_of_predictions(y_test, np.ones(n, dtype=int), retention_cost, empty_room_cost)},
            {"Strategy": f"Model (threshold {best_t})", **cost_of_predictions(y_test, y_pred, retention_cost, empty_room_cost)},
        ]
    )
    st.dataframe(comparison, hide_index=True)
    st.caption(
        f"The *Predict* page always uses the notebook's threshold ({artifact['threshold']}). "
        "Costs above are assumptions (default 1 : 10) and should be replaced with the hotel's real numbers."
    )

    # ----- Feature importance -----
    st.subheader("What drives the predictions?")
    imp = pd.Series(artifact["feature_importance"]).sort_values(ascending=False)
    imp.index = imp.index.str.replace(r"^(num|cat)__", "", regex=True)
    bar_chart(imp, "Top 15 features (LightGBM importance)", ylabel="Importance (split count)",
              horizontal=True, fmt="%.0f")
    st.caption("`adr` (room price) and `lead_time` are the strongest signals, in line with the EDA.")


# =====================================================================
# PAGE 4 - PREDICT
# =====================================================================
def option_box(label, column, default, key=None):
    options = sorted(df[column].unique().tolist())
    return st.selectbox(label, options, index=options.index(default), key=key)


def historical_rate(score, width=0.05):
    """Share of test-set bookings with a similar score that were really cancelled."""
    near = np.abs(artifact["y_prob_test"] - score) <= width
    return float(artifact["y_test"][near].mean()) if near.sum() >= 30 else None


def page_predict():
    st.title("🔮 Predict Cancellation Risk")
    st.write(
        f"Enter a new booking to get its cancellation **risk score**. "
        f"Bookings at or above **{artifact['threshold']:.2f}** are flagged as *at risk*."
    )
    tab_single, tab_batch = st.tabs(["Single booking", "Batch (CSV)"])

    # ----- Single booking -----
    with tab_single:
        with st.form("booking_form"):
            a, b, c = st.columns(3)
            with a:
                hotel = option_box("Hotel", "hotel", "City Hotel")
                lead_time = st.number_input("Lead time (days before arrival)", 0, 750, 100)
                month = st.selectbox("Arrival month", MONTHS, index=6)
                weekend_nights = st.number_input("Weekend nights", 0, 20, 1)
                week_nights = st.number_input("Week nights", 0, 50, 2)
                adults = st.number_input("Adults", 0, 10, 2)
                children = st.number_input("Children", 0, 10, 0)
                babies = st.number_input("Babies", 0, 10, 0)
            with b:
                meal = option_box("Meal plan", "meal", "BB")
                countries = df["country"].value_counts().index.tolist()
                country = st.selectbox("Country", countries, index=countries.index("PRT"))
                market_segment = option_box("Market segment", "market_segment", "Online TA")
                channel = option_box("Distribution channel", "distribution_channel", "TA/TO")
                room = option_box("Reserved room type", "reserved_room_type", "A")
                deposit = option_box("Deposit type", "deposit_type", "No Deposit")
                customer_type = option_box("Customer type", "customer_type", "Transient")
            with c:
                adr = st.number_input("Average daily rate (ADR)", 0.0, 4999.0, 100.0, 5.0)
                repeated = st.selectbox("Repeated guest?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                prev_cancel = st.number_input("Previous cancellations", 0, 30, 0)
                prev_ok = st.number_input("Previous bookings not cancelled", 0, 80, 0)
                changes = st.number_input("Booking changes", 0, 20, 0)
                waiting = st.number_input("Days in waiting list", 0, 400, 0)
                parking = st.number_input("Car parking spaces required", 0, 8, 0)
                requests = st.number_input("Special requests", 0, 5, 0)
            submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            if adults + children + babies == 0:
                st.error("A booking needs at least one guest.")
            else:
                booking = pd.DataFrame([{
                    "hotel": hotel, "lead_time": lead_time, "arrival_date_month": month,
                    "stays_in_weekend_nights": weekend_nights, "stays_in_week_nights": week_nights,
                    "adults": adults, "children": children, "babies": babies, "meal": meal,
                    "country": country, "market_segment": market_segment,
                    "distribution_channel": channel, "is_repeated_guest": repeated,
                    "previous_cancellations": prev_cancel, "previous_bookings_not_canceled": prev_ok,
                    "reserved_room_type": room, "booking_changes": changes, "deposit_type": deposit,
                    "days_in_waiting_list": waiting, "customer_type": customer_type, "adr": adr,
                    "required_car_parking_spaces": parking, "total_of_special_requests": requests,
                }])
                result = predict_risk(booking, artifact).iloc[0]
                score = float(result["risk_score"])
                at_risk = result["risk_label"] == "At risk of cancelling"
                actual = historical_rate(score)

                r1, r2, r3 = st.columns(3)
                r1.metric("Risk score", f"{score:.2f}")
                r2.metric("Decision", result["risk_label"])
                r3.metric("Actual cancel rate at similar scores", f"{actual:.0%}" if actual is not None else "n/a")
                st.progress(min(score, 1.0))
                if at_risk:
                    st.warning("Suggested action: reconfirm the booking, offer a small incentive, or plan for overbooking.")
                else:
                    st.success("Low risk: no special action needed.")
                st.caption(
                    f"The risk score is used for ranking, it is **not** a calibrated probability: the model was trained "
                    f"with class weighting, so scores run higher than real cancellation rates. The last metric shows "
                    f"what actually happened to test-set bookings with a similar score. The flag threshold "
                    f"({artifact['threshold']:.2f}) is intentionally low because a missed cancellation is assumed to cost "
                    f"~10x more than a wasted retention effort (see the Model page)."
                )

    # ----- Batch -----
    with tab_batch:
        st.write("Upload a CSV with the columns below (extra columns are ignored). Scores are added at the end.")
        template_cols = ["country" if c == "country_grp" else c for c in artifact["feature_columns"]]
        template = df[template_cols].sample(5, random_state=1)
        st.download_button("Download CSV template", template.to_csv(index=False),
                           file_name="booking_template.csv", mime="text/csv")

        uploaded = st.file_uploader("Bookings CSV", type="csv")
        if uploaded is not None:
            try:
                new_bookings = pd.read_csv(uploaded)
                scored = predict_risk(new_bookings, artifact)
            except Exception as e:
                st.error(f"Could not score this file: {e}")
            else:
                n_risk = int((scored["risk_label"] == "At risk of cancelling").sum())
                m1, m2, m3 = st.columns(3)
                m1.metric("Bookings scored", f"{len(scored):,}")
                m2.metric("At risk", f"{n_risk:,}")
                m3.metric("Share at risk", f"{n_risk / len(scored):.1%}")
                st.dataframe(scored.head(500))
                if len(scored) > 500:
                    st.caption("Showing the first 500 rows. Download the CSV for all rows.")
                st.download_button("Download scored CSV", scored.to_csv(index=False),
                                   file_name="scored_bookings.csv", mime="text/csv")


# =====================================================================
# NAVIGATION
# =====================================================================
pages = {
    "Overview": page_overview,
    "EDA": page_eda,
    "Model": page_model,
    "Predict": page_predict,
}
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", list(pages.keys()), label_visibility="collapsed")
st.sidebar.caption("Dataset: Hotel Booking Demand (119,390 bookings, 2015-2017).")
pages[choice]()
