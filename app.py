import streamlit as st
import requests
import pandas as pd
import base64

st.set_page_config(page_title="Stock Regime Analyzer", layout="wide")
st.title("📊 You technical Co-pilot")

# Input
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, MSFT):")

# On click
if st.button("Analyze"):
    if not ticker:
        st.warning("Please enter a valid stock ticker.")
    else:
        with st.spinner("Fetching and analyzing market regime..."):
            try:
                # Call FastAPI backend
                from regime_model import analyze_weekly_regime
                data = analyze_weekly_regime(ticker)

                # Header
                st.subheader(f"📌 Current Regime: **{data['current_regime']}**")
                if "action_plan" in data and data["action_plan"]:
                    st.markdown("### 🎯 Regime Action Plan")
                    st.markdown(f"**🔧 Action:** {data['action_plan']['action']}")
                    st.markdown(f"**🧭 Recommendation:** {data['action_plan']['recommendation']}")

                # Wyckoff Narrative
                st.markdown("---")
                st.subheader("🧠 Contextual Wyckoff Narrative")
                st.markdown(data.get("wyckoff_narrative", "_Not available_"))

                # Technical View
                st.subheader("📊 General Technical Analysis")
                st.markdown(data.get("technical_view", "_Not available_"))

                # Summary
                st.subheader("✅ Technical Insight Summary")
                st.markdown(data.get("summary", "_Not available_"))

                # Regime Plot
                st.markdown("---")
                st.subheader("📈 Regime Chart")
                st.image(base64.b64decode(data["image_base64"]), use_column_width=True)

                # Regime Table
                st.subheader("🗃️ Recent Regime Table")
                df = pd.DataFrame(data["regime_table"])
                df.columns = [col.capitalize() for col in df.columns]
                st.dataframe(df.tail(25), use_container_width=True)

            except Exception as e:
                st.error(f"❌ Failed to retrieve regime data.\nError: `{e}`")
