import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import numpy as np

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard | Retention Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root tokens ── */
:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --surface2:  #1a1e2a;
    --border:    #252836;
    --accent:    #5b8cff;
    --accent2:   #a78bfa;
    --danger:    #ff5c5c;
    --success:   #34d399;
    --warn:      #fbbf24;
    --text:      #e8eaf0;
    --muted:     #737891;
    --radius:    14px;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
.stApp { background: var(--bg); }
.block-container { padding: 2rem 3rem 4rem; max-width: 1280px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif; }

/* ── Hero header ── */
.hero-wrap {
    background: linear-gradient(135deg, #0d0f14 0%, #131a2e 60%, #0d0f14 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(91,140,255,.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -.5px;
    background: linear-gradient(90deg, #e8eaf0, #5b8cff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 .4rem;
}
.hero-sub {
    color: var(--muted);
    font-size: .95rem;
    font-weight: 300;
    margin: 0;
}

/* ── Section cards ── */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: .9rem;
}

/* ── Input styling ── */
.stNumberInput input,
.stSelectbox > div > div,
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSelectbox > div > div:hover,
.stNumberInput input:focus {
    border-color: var(--accent) !important;
}

/* ── Labels ── */
label, .stSelectbox label, .stNumberInput label {
    color: var(--muted) !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    letter-spacing: .3px !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: .5px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .85rem 2.4rem !important;
    width: 100% !important;
    transition: opacity .2s, transform .15s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
}

/* ── Verdict banner ── */
.verdict-churn {
    background: linear-gradient(135deg, rgba(255,92,92,.12), rgba(251,191,36,.06));
    border: 1px solid rgba(255,92,92,.35);
    border-left: 4px solid var(--danger);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
}
.verdict-stay {
    background: linear-gradient(135deg, rgba(52,211,153,.1), rgba(91,140,255,.06));
    border: 1px solid rgba(52,211,153,.3);
    border-left: 4px solid var(--success);
    border-radius: var(--radius);
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.4rem;
}
.verdict-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 0 0 .25rem;
}
.verdict-prob {
    font-size: .9rem;
    color: var(--muted);
    margin: 0;
}

/* ── Risk meter ── */
.risk-bar-wrap {
    background: var(--surface2);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: .6rem 0 .3rem;
}
.risk-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width .8s cubic-bezier(.4,0,.2,1);
}

/* ── Recommendation cards ── */
.rec-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: .75rem;
    display: flex;
    gap: .9rem;
    align-items: flex-start;
}
.rec-icon {
    font-size: 1.4rem;
    flex-shrink: 0;
    margin-top: .05rem;
}
.rec-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: .9rem;
    color: var(--text);
    margin-bottom: .2rem;
}
.rec-detail {
    font-size: .82rem;
    color: var(--muted);
    line-height: 1.5;
}
.rec-priority-high   { border-left: 3px solid var(--danger);  }
.rec-priority-medium { border-left: 3px solid var(--warn);    }
.rec-priority-low    { border-left: 3px solid var(--success); }

/* ── Factor chips ── */
.factor-grid {
    display: flex;
    flex-wrap: wrap;
    gap: .5rem;
    margin-top: .5rem;
}
.chip {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    font-size: .78rem;
    font-weight: 500;
    padding: .3rem .75rem;
    border-radius: 20px;
    white-space: nowrap;
}
.chip-risk {
    background: rgba(255,92,92,.12);
    border: 1px solid rgba(255,92,92,.3);
    color: #ff8080;
}
.chip-safe {
    background: rgba(52,211,153,.1);
    border: 1px solid rgba(52,211,153,.25);
    color: #5edfb0;
}

/* ── Metric tiles ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.metric-tile {
    flex: 1;
    min-width: 140px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: .2rem;
}
.metric-label {
    font-size: .73rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── SHAP table ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Info box ── */
.info-box {
    background: rgba(91,140,255,.08);
    border: 1px solid rgba(91,140,255,.22);
    border-radius: 10px;
    padding: .75rem 1rem;
    font-size: .83rem;
    color: #a0b4e8;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Model loader ────────────────────────────────────────────────────────────────
def bin_mapping(X):
    X = X.copy()
    for col in X.columns:
        X[col] = X[col].map({'Yes': 1, 'No': 0, 'No internet service': 0})
    return X

@st.cache_resource
def load_model():
    return joblib.load('churn_pipeline_v2.pkl')

try:
    loaded_model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.warning(f"⚠️ Model file not found: {e}. Running in demo mode.")


# ─── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <p class="hero-title">🛡️ ChurnGuard</p>
  <p class="hero-sub">Retention Intelligence Platform &nbsp;·&nbsp; SHAP-powered explainability &nbsp;·&nbsp; Actionable interventions</p>
</div>
""", unsafe_allow_html=True)


# ─── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Customer Profile</p>', unsafe_allow_html=True)

col_name, col_spacer = st.columns([1, 2])
with col_name:
    user_name = st.text_input("Customer Name", placeholder="e.g. Rohan Mehta")

st.markdown("<hr style='margin:.8rem 0'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

internet_options  = ["DSL", "Fiber optic", "No"]
yes_no_options    = ["Yes", "No", "No internet service"]
contract_options  = ["Month-to-month", "One year", "Two year"]
payment_options   = ["Electronic check", "Mailed check",
                     "Bank transfer (automatic)", "Credit card (automatic)"]

with col1:
    st.markdown('<p class="section-label" style="font-size:.6rem">Service Details</p>', unsafe_allow_html=True)
    internet_service  = st.selectbox("Internet Service", internet_options, index=1)
    online_security   = st.selectbox("Online Security",  yes_no_options,   index=1)
    online_backup     = st.selectbox("Online Backup",    yes_no_options,   index=1)

with col2:
    st.markdown('<p class="section-label" style="font-size:.6rem">Support & Protection</p>', unsafe_allow_html=True)
    device_protection = st.selectbox("Device Protection", yes_no_options, index=1)
    tech_support      = st.selectbox("Tech Support",      yes_no_options, index=1)
    contract          = st.selectbox("Contract Type",     contract_options, index=0)

with col3:
    st.markdown('<p class="section-label" style="font-size:.6rem">Billing</p>', unsafe_allow_html=True)
    payment_method    = st.selectbox("Payment Method", payment_options, index=0)
    tenure            = st.number_input("Tenure (months)", min_value=0, value=6, step=1)
    monthly_charges   = st.number_input("Monthly Charges ($)", min_value=0.0, value=85.0, format="%.2f")

total_charges = float(tenure * monthly_charges)
st.markdown(f'<div class="info-box">📊 Calculated Total Charges: <strong>${total_charges:,.2f}</strong> &nbsp;(tenure × monthly rate)</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("🔍  Analyse Churn Risk")


# ─── Helper: SHAP-based recommendations ─────────────────────────────────────────
def generate_recommendations(shap_df, churn_prob):
    """
    Returns a list of dicts: {icon, title, detail, priority}
    based on top positive-SHAP features (those pushing toward churn).
    """
    recs = []
    top_risk = shap_df[shap_df['Impact'] > 0].head(6)

    for _, row in top_risk.iterrows():
        feat  = row['Feature']
        impact = row['Impact']
        priority = "high" if impact > 0.15 else ("medium" if impact > 0.07 else "low")

        # ── Contract ──────────────────────────────────────────────────────────
        if "Contract_Month-to-month" in feat:
            recs.append({
                "icon": "📋",
                "title": "Upgrade to Annual/Bi-Annual Contract",
                "detail": (
                    "Month-to-month customers churn 3-5× more than long-term subscribers. "
                    "Offer a 10–15% discount or a free month to incentivize a 1-year switch. "
                    "Highlight cost savings in the outreach email."
                ),
                "priority": priority
            })

        # ── Payment: electronic check ────────────────────────────────────────
        elif "PaymentMethod_Electronic check" in feat:
            recs.append({
                "icon": "💳",
                "title": "Migrate to Auto-Pay",
                "detail": (
                    "Electronic-check users are significantly more likely to churn—often due to "
                    "friction at each billing cycle. Offer a $5/month loyalty discount for switching "
                    "to bank transfer or credit card auto-pay."
                ),
                "priority": priority
            })

        # ── Fiber optic (high expectations) ─────────────────────────────────
        elif "InternetService_Fiber optic" in feat:
            recs.append({
                "icon": "🌐",
                "title": "Proactive Fiber Quality Check",
                "detail": (
                    "Fiber optic subscribers have the highest expectations and churn faster when "
                    "service quality dips. Schedule a proactive health-check call, resolve any "
                    "latency complaints, and offer a speed upgrade trial."
                ),
                "priority": priority
            })

        # ── No Tech Support ──────────────────────────────────────────────────
        elif feat == "TechSupport":
            recs.append({
                "icon": "🛠️",
                "title": "Enrol in Tech Support Plan",
                "detail": (
                    "Customers without tech support are more likely to leave after unresolved issues. "
                    "Offer a 60-day free tech-support trial to demonstrate value and build stickiness."
                ),
                "priority": priority
            })

        # ── No Online Security ───────────────────────────────────────────────
        elif feat == "OnlineSecurity":
            recs.append({
                "icon": "🔒",
                "title": "Bundle Online Security Add-On",
                "detail": (
                    "Customers without security add-ons feel less invested in the platform. "
                    "A complimentary 3-month security bundle trial can increase perceived value "
                    "and reduce cancellation intent."
                ),
                "priority": priority
            })

        # ── No Online Backup ─────────────────────────────────────────────────
        elif feat == "OnlineBackup":
            recs.append({
                "icon": "☁️",
                "title": "Activate Online Backup Trial",
                "detail": (
                    "Add-on services like backup increase switching costs. "
                    "Offer a free 2-month backup trial—customers who use it rarely churn."
                ),
                "priority": priority
            })

        # ── No Device Protection ─────────────────────────────────────────────
        elif feat == "DeviceProtection":
            recs.append({
                "icon": "🖥️",
                "title": "Offer Device Protection Plan",
                "detail": (
                    "Customers without device protection have less financial lock-in. "
                    "Pitch a bundled protection plan at a discount to increase product stickiness."
                ),
                "priority": priority
            })

        # ── High monthly charges ─────────────────────────────────────────────
        elif feat == "MonthlyCharges":
            recs.append({
                "icon": "💰",
                "title": "Personalised Price Review",
                "detail": (
                    "High monthly charges are a leading churn trigger. "
                    "Proactively reach out with a tailored loyalty discount (5–10%) or "
                    "repackage the plan to remove underused services, reducing perceived cost."
                ),
                "priority": priority
            })

        # ── Low tenure ───────────────────────────────────────────────────────
        elif feat == "tenure":
            recs.append({
                "icon": "🌱",
                "title": "New-Customer Onboarding Programme",
                "detail": (
                    "Low-tenure customers are high-risk—they haven't built habit or loyalty yet. "
                    "Enrol in a 90-day onboarding journey with milestone rewards, a dedicated "
                    "onboarding call, and a first-90-days satisfaction guarantee."
                ),
                "priority": priority
            })

    # Deduplicate by title
    seen = set()
    unique_recs = []
    for r in recs:
        if r['title'] not in seen:
            seen.add(r['title'])
            unique_recs.append(r)

    # Sort by priority
    order = {"high": 0, "medium": 1, "low": 2}
    unique_recs.sort(key=lambda x: order[x['priority']])
    return unique_recs[:5]  # top 5


# ─── Prediction ──────────────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "tenure", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "Contract", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

if predict_clicked:
    user_data = {
        "tenure": tenure,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "Contract": contract,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    if not model_loaded:
        # ── Demo mode ──────────────────────────────────────────────────────
        st.warning("Running in demo mode — model not loaded.")

    else:
        with st.spinner("Running SHAP analysis…"):
            input_df = pd.DataFrame([user_data])
            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')

            preprocessor = loaded_model.named_steps['preprocessor']
            model        = loaded_model.named_steps['classifier']

            X_transformed    = preprocessor.transform(input_df[EXPECTED_COLUMNS])
            prediction       = loaded_model.predict(input_df[EXPECTED_COLUMNS])
            probability      = loaded_model.predict_proba(input_df[EXPECTED_COLUMNS])
            churn_prob       = probability[0][1]

            # Feature names
            bin_cols = ['TechSupport', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection']
            num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            cat_encoder      = preprocessor.named_transformers_['cat']
            cat_cols         = ['Contract', 'PaymentMethod', 'InternetService']
            cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
            feature_names    = list(bin_cols) + list(num_cols) + list(cat_feature_names)

            explainer  = shap.Explainer(model)
            shap_data  = explainer(X_transformed)

        # ── Build SHAP dataframe ───────────────────────────────────────────
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Impact':  shap_data.values[0]
        }).sort_values(by='Impact', key=abs, ascending=False)

        # ── Risk level label ──────────────────────────────────────────────
        if churn_prob >= 0.70:
            risk_label, risk_color = "High Risk",   "#ff5c5c"
        elif churn_prob >= 0.40:
            risk_label, risk_color = "Medium Risk",  "#fbbf24"
        else:
            risk_label, risk_color = "Low Risk",    "#34d399"

        display_name = user_name.strip() if user_name.strip() else "This customer"

        # ── Verdict banner ────────────────────────────────────────────────
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="verdict-churn">
                <p class="verdict-title" style="color:var(--danger)">
                    ⚠️ {display_name} is likely to churn
                </p>
                <p class="verdict-prob">
                    Churn probability: <strong style="color:var(--danger)">{churn_prob:.1%}</strong>
                    &nbsp;·&nbsp; Risk tier: <strong style="color:{risk_color}">{risk_label}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-stay">
                <p class="verdict-title" style="color:var(--success)">
                    ✅ {display_name} is likely to stay
                </p>
                <p class="verdict-prob">
                    Churn probability: <strong style="color:var(--success)">{churn_prob:.1%}</strong>
                    &nbsp;·&nbsp; Risk tier: <strong style="color:{risk_color}">{risk_label}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Risk bar ──────────────────────────────────────────────────────
        bar_pct = int(churn_prob * 100)
        st.markdown(f"""
        <div style="margin-bottom:1.4rem">
            <div style="display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted);margin-bottom:.3rem">
                <span>Retention Confidence</span><span>{100-bar_pct}%</span>
            </div>
            <div class="risk-bar-wrap">
                <div class="risk-bar-fill"
                     style="width:{bar_pct}%;background:linear-gradient(90deg,{risk_color}88,{risk_color})">
                </div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:.78rem;color:var(--muted)">
                <span>0% churn risk</span><span>100% churn risk — <strong style="color:{risk_color}">{bar_pct}%</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric tiles ──────────────────────────────────────────────────
        top_driver   = shap_df.iloc[0]['Feature'].replace("_", " ").replace("Contract ", "")
        n_risk_factors = len(shap_df[shap_df['Impact'] > 0.05])

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-value" style="color:{risk_color}">{churn_prob:.0%}</div>
                <div class="metric-label">Churn Probability</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value" style="color:var(--accent2)">{n_risk_factors}</div>
                <div class="metric-label">Active Risk Signals</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value" style="color:var(--warn);font-size:1rem;padding-top:.4rem">{top_driver[:18]}</div>
                <div class="metric-label">Top Driver</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value" style="color:var(--text)">{tenure}</div>
                <div class="metric-label">Months Tenure</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column layout: Recommendations | SHAP ─────────────────────
        left, right = st.columns([1.05, 1], gap="large")

        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">Recommended Interventions</p>', unsafe_allow_html=True)

            recs = generate_recommendations(shap_df, churn_prob)
            if recs:
                for r in recs:
                    priority_class = f"rec-priority-{r['priority']}"
                    priority_badge_color = {"high": "#ff5c5c", "medium": "#fbbf24", "low": "#34d399"}[r['priority']]
                    st.markdown(f"""
                    <div class="rec-card {priority_class}">
                        <div class="rec-icon">{r['icon']}</div>
                        <div>
                            <div class="rec-title">{r['title']}
                                <span style="font-size:.68rem;font-weight:400;
                                      color:{priority_badge_color};
                                      background:rgba(255,255,255,.05);
                                      border:1px solid {priority_badge_color}44;
                                      border-radius:20px;padding:.1rem .5rem;
                                      margin-left:.5rem;text-transform:uppercase;
                                      letter-spacing:.8px">{r['priority']}</span>
                            </div>
                            <div class="rec-detail">{r['detail']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:var(--muted);font-size:.88rem">No high-impact risk factors detected. Customer profile is healthy.</p>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">SHAP Feature Impact</p>', unsafe_allow_html=True)

            # Factor chips
            risk_feats = shap_df[shap_df['Impact'] >  0.05].head(4)
            safe_feats = shap_df[shap_df['Impact'] < -0.05].head(4)
            chips_html = '<div class="factor-grid">'
            for _, r in risk_feats.iterrows():
                label = r['Feature'].replace("_", " ").replace("Contract ", "")[:22]
                chips_html += f'<span class="chip chip-risk">▲ {label}</span>'
            for _, r in safe_feats.iterrows():
                label = r['Feature'].replace("_", " ").replace("Contract ", "")[:22]
                chips_html += f'<span class="chip chip-safe">▼ {label}</span>'
            chips_html += '</div>'
            st.markdown(chips_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Waterfall chart with dark theme
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('#13161e')
            ax.set_facecolor('#13161e')

            shap.plots.waterfall(shap_data[0], show=False, max_display=10)
            plt.gcf().set_facecolor('#13161e')
            for spine in plt.gca().spines.values():
                spine.set_edgecolor('#252836')
            plt.tick_params(colors='#737891')
            plt.tight_layout(pad=1.2)
            st.pyplot(plt.gcf())
            plt.close('all')

            st.markdown('</div>', unsafe_allow_html=True)

        # ── Detailed SHAP table (collapsible) ─────────────────────────────
        with st.expander("📋 Full SHAP Impact Table", expanded=False):
            styled_df = shap_df.copy()
            styled_df['Direction'] = styled_df['Impact'].apply(
                lambda v: "🔺 Increases churn" if v > 0 else "🔻 Reduces churn"
            )
            styled_df['Impact (abs)'] = styled_df['Impact'].abs().round(4)
            styled_df['Impact']       = styled_df['Impact'].round(4)
            st.dataframe(
                styled_df[['Feature', 'Impact', 'Impact (abs)', 'Direction']],
                use_container_width=True,
                hide_index=True
            )