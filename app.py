import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Will This Visitor Buy?", page_icon="🛒", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load("purchase_intent_model.joblib")


bundle = load_model()
pipe, threshold, defaults = bundle["pipeline"], bundle["threshold"], bundle["defaults"]
num_def, cat_def = defaults["num"], defaults["cat"]

st.title("🛒 Will This Visitor Buy?")
st.write("Enter what a visitor did in their browsing session. The model estimates the "
         "probability of a purchase and suggests an action for the website.")

# ---- Sidebar: business policy ----
st.sidebar.header("Action policy")
high = st.sidebar.slider("No action above (will buy anyway)", 0.30, 0.95, 0.60, 0.05)
low = st.sidebar.slider("Coupon above", 0.05, 0.60, float(min(max(threshold, 0.05), 0.6)), 0.05,
                        help="Defaults to the model's tuned threshold.")
chat_exit = st.sidebar.slider("Open chat if exit rate is above", 0.00, 0.20, 0.05, 0.01)
st.sidebar.caption("These thresholds are business assumptions, not model outputs. "
                   "Tune them with the cost of a coupon vs. the value of a sale.")

# ---- Form ----
with st.form("session"):
    st.subheader("Browsing session")
    c1, c2 = st.columns(2)
    product_pages = c1.number_input("Product pages visited", 0, 1000, 10)
    product_time = c2.number_input("Time on product pages (seconds)", 0, 100000, 400)
    admin_pages = c1.number_input("Account/admin pages visited", 0, 100, 1)
    admin_time = c2.number_input("Time on admin pages (seconds)", 0, 20000, 30)
    info_pages = c1.number_input("Informational pages visited", 0, 100, 0)
    info_time = c2.number_input("Time on informational pages (seconds)", 0, 20000, 0)

    c3, c4, c5 = st.columns(3)
    bounce = c3.number_input("Bounce rate", 0.0, 0.2, 0.02, 0.005, format="%.3f")
    exit_rate = c4.number_input("Exit rate", 0.0, 0.2, 0.04, 0.005, format="%.3f")
    page_value = c5.number_input("Page value", 0.0, 500.0, 0.0, 1.0,
                                 help="Average value of the pages seen. 0 means no revenue-linked page yet.")

    c6, c7, c8 = st.columns(3)
    month = c6.selectbox("Month", defaults["months"])
    visitor = c7.selectbox("Visitor type", defaults["visitor_types"])
    weekend = c8.checkbox("Weekend")
    special = st.select_slider("Closeness to a special day (0 = far, 1 = the day itself)",
                               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], 0.0)

    with st.expander("Technical details (optional)"):
        t1, t2 = st.columns(2)
        os_ = t1.text_input("Operating system code", cat_def["OperatingSystems"])
        browser = t2.text_input("Browser code", cat_def["Browser"])
        region = t1.text_input("Region code", cat_def["Region"])
        traffic = t2.text_input("Traffic type code", cat_def["TrafficType"])

    go = st.form_submit_button("Predict", type="primary")

if go:
    row = pd.DataFrame([{
        "Administrative": admin_pages, "Administrative_Duration": admin_time,
        "Informational": info_pages, "Informational_Duration": info_time,
        "ProductRelated": product_pages, "ProductRelated_Duration": product_time,
        "BounceRates": bounce, "ExitRates": exit_rate, "PageValues": page_value,
        "SpecialDay": special, "Month": month, "OperatingSystems": os_, "Browser": browser,
        "Region": region, "TrafficType": traffic, "VisitorType": visitor, "Weekend": int(weekend),
    }])[bundle["columns"]]
    try:
        p = float(pipe.predict_proba(row)[0, 1])
    except Exception as e:
        st.error(f"Could not score this session. Please check the inputs. ({e})")
        st.stop()

    st.metric("Purchase probability", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))

    if p >= high:
        st.success("**No action.** This visitor is very likely to buy anyway. Don't give a discount.")
    elif p >= low:
        st.warning("**Show a coupon.** Likely buyer who may still hesitate; a small incentive can tip the decision.")
    elif exit_rate >= chat_exit:
        st.info("**Open the chat assistant.** Low purchase chance and about to leave; "
                "a helpful answer may save the session.")
    else:
        st.write("**No action.** Low purchase probability and no strong sign of leaving.")

    st.caption(f"Model: {bundle['model_name']} · decision threshold {threshold:.2f}. "
               "A prediction is a probability, not a guarantee.")
