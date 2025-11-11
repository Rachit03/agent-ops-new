import streamlit as st
import pandas as pd
from queryController import LLMQueries
from modules import *
# from modules import build_agent_tool_mermaid,getAgentData,getagentLLMToolmapping,prepGraphData,process_trace_data,calcCost,parse_agent_tool_latency,extract_agent_data
import altair as alt


st.set_page_config(page_title="Agent Performance", layout="wide")

st.title("Agent Performance")

llm = LLMQueries()

with st.expander("Filters",expanded=True):
# --- Filters Box ---
    col1, col2, col3, col4 = st.columns([0.4, 0.2, 0.2, 0.2],vertical_alignment="center")
    with col1:
        selected = st.selectbox("Projects", llm.get_dropdown_options(), label_visibility="visible")
    with col2:
        min_date, max_date = llm.get_date_range_for_project(selected)
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, label_visibility="visible")
    with col3:
        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, label_visibility="visible")
    with col4:
        st.markdown("&nbsp;", unsafe_allow_html=True)

# --- Metric Section ---
val1 = llm.get_tile1_value(selected, start_date, end_date)
val2 = round(llm.get_tile2_value(selected, start_date, end_date), 2)
val3 = llm.get_tile3_value(selected, start_date, end_date)
val4 = llm.get_tile4_value(selected, start_date, end_date)


try:
    # Automatically call the external function using current filters
    df_agent = llm.get_agent_score_all_data(selected, start_date, end_date)
    summarized_df = llm.sumarize_agent_score(df_agent)

except Exception as e:
    st.error(f"⚠️ Error while loading custom data: {e}")

# Ensure datetime conversion
df_agent["start_time"] = pd.to_datetime(df_agent["start_time"])

# Select trend period
period = st.selectbox("Select Trend Period", ["Weekly", "Monthly"])

# --- Create period column ---
if period == "Weekly":
    df_agent["period"] = df_agent["start_time"].dt.to_period("W").apply(lambda r: r.start_time)
    df_agent["year"] = df_agent["start_time"].dt.year
    df_agent["week"] = df_agent["start_time"].dt.isocalendar().week
    # df_agent["period_label"] = df_agent["week"]
    df_agent["period_label"] = df_agent["year"].astype(str) + "-W" + df_agent["week"].astype(str)
    period_col = "period_label"
else:
    df_agent["period_label"] = df_agent["start_time"].dt.to_period("M").apply(lambda r: r.start_time)
    period_col = "period_label"

# --- Compute agent score per period ---
results = []

for p, subset in df_agent.groupby(period_col):
    score_df_agent = llm.sumarize_agent_score(subset)  # summarized df_agent per period
    # score_df_agent["period"] = p
    score_df_agent["period_label"] = p
    results.append(score_df_agent)

# Combine all period results
trend_df = pd.concat(results, ignore_index=True)
if period == "Weekly":
    trend_df["year"] = trend_df["period_label"].str.extract(r"(\d{4})").astype(int)
    trend_df["week"] = trend_df["period_label"].str.extract(r"W(\d{1,2})").astype(int)
    trend_df = trend_df.sort_values(["year", "week"])
    # trend_df = trend_df.sort_values(["period_label"])
    trend_df["period_label"] = trend_df["period_label"].astype("str")
    # x_field = "period_label"
    # x_title = "Week Start"
else:
    trend_df["period_label"] = pd.to_datetime(trend_df["period_label"], format="%Y-%m")
    trend_df = trend_df.sort_values("period_label")
    trend_df["period_label"] = trend_df["period_label"].dt.strftime("%b %Y")
    # x_field = "period_label:T"
    # x_title = "Month"


print(trend_df)
# --- Visualization ---

col1, col2 = st.columns(2)

with col1:
    # st.subheader(f"All Agents Trend ({period}) ")
    st.markdown(
    "### All Agents Trend",
    help="This chart shows the average agent LLM score for all agents by week/month."
)
    line_chart = (
        alt.Chart(trend_df)
        .mark_line(point=True)
        .encode(
            # x=alt.X("period_label", title="Time Period", axis=alt.Axis(labelAngle=-45)),
            x=alt.X(
            "period_label:N",
            title="Time Period",
            sort=list(trend_df["period_label"].unique()),  # enforce sort order
            axis=alt.Axis(labelAngle=-45)
        ),

            y=alt.Y("avg_llm_score:Q", title="Agent Eval Score"),
            color=alt.Color("agent_name:N", title="Agent Name"),
            detail="agent_name:N",
            tooltip=["period_label", "agent_name", "avg_llm_score"]
        )
        # .properties(title=f"Agent Score Trend ({period})", width=800, height=400)
    )

    st.altair_chart(line_chart, use_container_width=True)

with col2:
    st.markdown(

    "### Single Agents Trend",
    help="This chart shows the average normalized score for an agents by week/month based on LLM Eval Scire, latency, token used and error rate."
)

    # st.subheader(f"Single Agent Trend ({period})")
    agent_list = sorted(trend_df["agent_name"].unique())
    default_agent = agent_list[0] if agent_list else None

    selected_agent = st.selectbox("Select an Agent to view specific trend:", agent_list, index=0)

    if selected_agent:
        filtered_df = trend_df[trend_df["agent_name"] == selected_agent]

        single_agent_chart = (
            alt.Chart(filtered_df)
            .mark_line(point=True, color="#0072B2")
            .encode(
                # x=alt.X("period_label", title="Time Period", axis=alt.Axis(labelAngle=-45)),
                x=alt.X(
            "period_label:N",
            title="Time Period",
            sort=list(trend_df["period_label"].unique()),  # enforce sort order
            axis=alt.Axis(labelAngle=-45)
        ),

                y=alt.Y("normalized_score:Q", title="Agent Score"),
                tooltip=["agent_name", "normalized_score", "period_label"]
            )
            # .properties(title=f"{selected_agent} Trend ({period})", width=800, height=300)
        )

        st.altair_chart(single_agent_chart, use_container_width=True)

if isinstance(summarized_df, pd.DataFrame) and not summarized_df.empty:
    st.dataframe(summarized_df, use_container_width=True, hide_index=True, height=300)

else:
    st.info("No data returned from the custom function for the selected filters.")
