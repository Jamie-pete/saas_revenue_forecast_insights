import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import openai


st.set_page_config(page_title="SaaS Revenue Forecasting & Business Advisor",
                   layout="wide")

st.title("📊 SaaS Revenue Forecasting & Business Advisor")


@st.cache_data
def load_data():
    mrr_clean = pd.read_csv(
        "/Users/james_peter/Documents/saas_revenue_forecast_insights/data/processed/mrr_clean.csv",
        parse_dates=["date"]
    )
    mrr_forecast = pd.read_csv(
        "/Users/james_peter/Documents/saas_revenue_forecast_insights/data/processed/mrr_forecast.csv",
        parse_dates=["date"]
    )
    trans_cust = pd.read_csv(
        "/Users/james_peter/Documents/saas_revenue_forecast_insights/data/processed/trans_cust_clean.csv",
        parse_dates=["date", "signup_date", "churn_date"]
    )
    return mrr_clean, mrr_forecast, trans_cust

mrr_clean, mrr_forecast, trans_cust = load_data()

mrr_clean.columns = mrr_clean.columns.str.lower()


tab1, tab2, tab3 = st.tabs(["Revenue Dashboard", "Forecasting", "Business Insights"])


# TAB 1: REVENUE DASHBOARD

with tab1:
    st.subheader("Monthly Recurring Revenue (Mrr) Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=mrr_clean, x="date", y="mrr", ax=ax, marker="o")
    ax.set_title("MrrOver Time")
    ax.set_ylabel("Mrr ($)")
    st.pyplot(fig)

    # Basic stats
    latest_mrr = mrr_clean["mrr"].iloc[-1]
    yoy_growth = ((latest_mrr - mrr_clean["mrr"].iloc[0]) / mrr_clean["mrr"].iloc[0]) * 100
    st.metric("Latest mrr", f"${latest_mrr:,.0f}", f"{yoy_growth:.1f}% growth since start")

# TAB 2: FORECASTING

with tab2:
    st.subheader("MRR Forecast")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=mrr_clean, x="date", y="mrr", ax=ax, label="Actual MRR", color="blue")
    sns.lineplot(data=mrr_forecast, x="date", y="mrr_forecast", ax=ax, label="Forecast", color="orange")
    ax.fill_between(
        mrr_forecast["date"],
        mrr_forecast["lower_ci"],
        mrr_forecast["upper_ci"],
        color="orange",
        alpha=0.2,
        label="Confidence Interval"
    )
    ax.set_title("Actual vs Forecasted MRR with Confidence Interval")
    ax.set_xlabel("Date")
    ax.set_ylabel("MRR")
    ax.legend()
    st.pyplot(fig)

    #Interactive Forecast Button
    st.markdown("### 🔮 Generate Future Forecast")
    steps = st.slider("Select months to forecast:", 3, 24, 12)  
    if st.button("Run Forecast"):
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(mrr_clean["mrr"], order=(1,1,1))  # simple ARIMA(1,1,1)
        model_fit = model.fit()

        forecast = model_fit.get_forecast(steps=steps)
        forecast_index = pd.date_range(
            start=mrr_clean["date"].iloc[-1] + pd.offsets.MonthBegin(),
            periods=steps,
            freq="MS"
        )
        forecast_df = pd.DataFrame({
            "date": forecast_index,
            "forecast": forecast.predicted_mean,
            "lower_ci": forecast.conf_int().iloc[:, 0],
            "upper_ci": forecast.conf_int().iloc[:, 1]
        })

        #forecast result
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=mrr_clean, x="date", y="mrr", ax=ax2, label="Historical MRR", color="blue")
        sns.lineplot(data=forecast_df, x="date", y="forecast", ax=ax2, label="Forecast", color="green")
        ax2.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color="green", alpha=0.2)
        ax2.set_title(f"Dynamic Forecast for Next {steps} Months")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("MRR")
        ax2.legend()
        st.pyplot(fig2)


    st.markdown("### 📊 ARIMA Model Diagnostics")
    st.write("These charts help understand how well ARIMA fits the data.")

    model = ARIMA(mrr_clean["mrr"], order=(1,1,1))
    model_fit = model.fit()

    # Residuals plot
    residuals = model_fit.resid
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    sns.lineplot(x=mrr_clean["date"], y=residuals, ax=ax3, color="red")
    ax3.axhline(0, linestyle="--", color="black")
    ax3.set_title("Residuals over Time (Should hover around 0)")
    st.pyplot(fig3)

    # Histogram of residuals
    fig4, ax4 = plt.subplots(figsize=(6, 3))
    sns.histplot(residuals, bins=20, kde=True, ax=ax4, color="purple")
    ax4.set_title("Distribution of Residuals (Check Normality)")
    st.pyplot(fig4)

    st.markdown("### 📝 Interpretation")
    st.write("""
    - **Blue Line** = Actual MRR trend observed.  
    - **Orange Line** = ARIMA forecast with confidence interval.  
    - **Residual Plots** help verify that the model errors are random.  
    - **Confidence Interval** shows the likely range of future MRR.  

    ✅ If residuals look random & centered at 0 → Good model fit.  
    ⚠️ If residuals show trend/pattern → Model may need tuning (try different ARIMA orders).
    """)


# TAB 3: BUSINESS INSIGHTS (AUTO)
with tab3:
    st.subheader("📊 Business Insights (Auto-Generated)")

    def generate_business_insights(df, forecast_df, customers):
        try:
            insights = []

            #MRR STATS 
            latest_mrr = df['mrr'].iloc[-1] if 'mrr' in df.columns and not df.empty else None
            yoy_growth = ((df['mrr'].iloc[-1] - df['mrr'].iloc[0]) / df['mrr'].iloc[0]) if not df.empty else None
            avg_growth = forecast_df['mrr_forecast'].pct_change().mean() if 'mrr_forecast' in forecast_df.columns else None

            latest_mrr_text = f"${latest_mrr:,.2f}" if latest_mrr else "N/A"
            yoy_growth_text = f"{yoy_growth:.2%}" if yoy_growth else "N/A"
            avg_growth_text = f"{avg_growth:.2%}" if avg_growth else "N/A"

            insights.append(f"💰 Latest recorded MRR: **{latest_mrr_text}**")
            insights.append(f"📈 Your growth since start: **{yoy_growth_text}**")
            insights.append(f"📊 Forecasted average monthly growth: **{avg_growth_text}**")

            #CHURN ANALYSIS 
            churn_rate = None
            if 'churn_date' in customers.columns:
                churned = customers[customers['churn_date'].notna()]
                churn_rate = len(churned) / len(customers) if len(customers) > 0 else None
                churn_text = f"{churn_rate:.2%}" if churn_rate else "N/A"
                insights.append(f"⚠️ Historical churn rate: **{churn_text}**")
                if churn_rate and churn_rate > 0.05:
                    insights.append("🚨 High churn detected – focus on retention strategies.")
                else:
                    insights.append("✅ Churn rate under control.")

            #ACTIONABLE RECS 
            if avg_growth and avg_growth > 0.03:
                insights.append("📢 Recommendation: Growth trend is strong, double down on scaling acquisition.")
            elif avg_growth and avg_growth < 0:
                insights.append("🛑 Revenue contraction expected. Audit churn drivers and product stickiness.")
            else:
                insights.append("🔍 Explore upsell/cross-sell opportunities to boost flat growth.")

            return "\n\n".join(insights)

        except Exception as e:
            return f"Error generating insights: {e}"

    # Display insights
    insights = generate_business_insights(mrr_clean, mrr_forecast, trans_cust)
    st.markdown(insights)

    #VISUALIZATIONS 
    st.subheader("📉 Customer Acquisition vs Churn")
    cust_summary = trans_cust.copy()
    cust_summary['month'] = cust_summary['date'].dt.to_period("M")
    churned = cust_summary[cust_summary['churn_date'].notna()].groupby('month').size()
    new_signups = cust_summary.groupby('month')['signup_date'].count()

    churn_vs_signup = pd.DataFrame({
        "New Signups": new_signups,
        "Churned Customers": churned
    }).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    churn_vs_signup.plot(kind="bar", ax=ax)
    ax.set_title("New Signups vs Churned Customers (Monthly)")
    ax.set_ylabel("Customers")
    st.pyplot(fig)

    st.subheader("📈 Actual vs Forecasted MRR (Snapshot)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=mrr_clean, x="date", y="mrr", ax=ax2, label="Actual MRR", color="blue")
    sns.lineplot(data=mrr_forecast, x="date", y="mrr_forecast", ax=ax2, label="Forecasted MRR", color="orange")
    ax2.set_title("MRR Trend with Forecast Overlay")
    ax2.set_ylabel("MRR ($)")
    ax2.legend()
    st.pyplot(fig2)
