import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# output directory and fienames
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
BASE_DIR = PROJECT_DIR / "outputs"
DEFAULT_SUBDIR = "run1975"
DEFAULT_CHAN_FILE = "metrics.parquet"
DEFAULT_PULSE_FILE = "pulses.parquet"


@st.cache_data
def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found at {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    return df


def main():

    # titles and layout 
    st.set_page_config(
        page_title="Anomaly Dashboard",
        layout="wide",
    )


    st.title("Detector Anomaly Dashboard")
    st.caption("Monitoring reconstruction error per channel from the autoencoder.")

    # Sidebar, data source and filtering 
    st.sidebar.header("Data source")
    subdir = st.sidebar.text_input(
        "Output subfolder (relative to outputs/)",
        DEFAULT_SUBDIR
    )
    chan_file = st.sidebar.text_input(
        "Channel metrics filename",
        DEFAULT_CHAN_FILE
    )
    pulse_file = st.sidebar.text_input(
        "Pulse metrics filename",
        DEFAULT_PULSE_FILE
    )

    metrics_path = BASE_DIR / subdir / chan_file
    pulse_metrics_path = BASE_DIR / subdir / pulse_file

    try:
        df = load_metrics(metrics_path)
        df_pulses = load_metrics(pulse_metrics_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    if df.empty:
        st.warning("Channel file is empty.")
        st.stop()

    if df_pulses.empty:
        st.warning("Pulse-level metrics file is empty.")
        st.stop()

    with st.sidebar.expander("Channel DF columns"):
        st.write(list(df.columns))
    with st.sidebar.expander("Pulse DF columns"):
        st.write(list(df_pulses.columns))

    st.sidebar.header("Filters")

    if "channel" not in df.columns:
        st.error("No 'channel' column in channel metrics.")
        st.stop()
    if "mean_err" not in df.columns:
        st.error("No 'mean_err' column in channel metrics.")
        st.stop()
    if "rating" not in df.columns:
        st.error("No 'rating' column in channel metrics.")
        st.stop()

    all_channels = sorted(df["channel"].unique())
    channel_options = ["All"] + all_channels

    selected = st.sidebar.multiselect(
        "Channels",
        options=channel_options,
        default="All"
    )

    if "All" in selected or selected == []:
        selected_channels = all_channels
    else:
        selected_channels = selected


    loss_min = float(df["mean_err"].min())
    loss_max = float(df["mean_err"].max())
    default_thresh = float(df["mean_err"].quantile(0.95))


    mask = df["channel"].isin(selected_channels)
    fdf = df[mask].copy()

    if fdf.empty:
        st.warning("No data for chosen filters.")
        st.stop()

    pulse_mask = df_pulses["channel"].isin(selected_channels)
    fdf_pulses = df_pulses[pulse_mask].copy()

    if fdf_pulses.empty:
        st.warning("No pulse-level data for chosen filters.")

    tab_chan, tab_pulse = st.tabs(["Channel overview", "Pulse-level view"])


    with tab_chan:
        st.subheader("Channel-level anomaly summary")
        k1, k2 = st.columns(2)

        n_channels = fdf["channel"].nunique()
        n_high = (fdf["rating"].str.lower() == "high").sum()

        if not fdf_pulses.empty:
            total_pulses = int(len(fdf_pulses))
        else:
            total_pulses = 0

        with k1:
            st.metric(
                "High-anomaly channels",
                f"{n_high} / {n_channels}",
                help="Number of channels with Rating = 'High'."
            )
        with k2:
            st.metric(
                "Total pulses (selected channels)",
                f"{total_pulses}",
                help="Number of pulses in the pulse-level dataset for selected channels."
            )

        st.markdown("---")


        st.subheader("Channel severity breakdown")
        left, right = st.columns([1, 2])


        with left:

            severity_counts = (
                fdf["rating"]
                .value_counts()
                .reindex(["High", "Medium", "Low"])
                .fillna(0)
            )
            severity_df = severity_counts.reset_index()
            severity_df.columns = ["rating", "count"]

            fig_donut = px.pie(
                severity_df,
                names="rating",
                values="count",
                hole=0.65,
                title="Channels by anomaly rating",
                color="rating",
                color_discrete_map={
                    "High": "#ff4b4b",
                    "Medium": "#ffb703",
                    "Low": "#38b000"
                },
            )
            fig_donut.update_layout(template="plotly_dark")
            st.plotly_chart(fig_donut, use_container_width=True)


        with right:

            fig_dist = go.Figure()

            fig_dist.add_trace(
                go.Histogram(
                    x=fdf["mean_err"],
                    nbinsx=30,
                    marker=dict(
                        color="rgba(0, 150, 255, 0.75)",
                        line=dict(color="rgba(0, 150, 255, 1)", width=1.5)
                    ),
                    opacity=0.85,
                    name="Mean RL",
                )
            )

            fig_dist.update_layout(
                title="Channel Mean RL Distribution",
                xaxis_title="Mean RL (Reconstruction Loss)",
                yaxis_title="Channel Count",
                bargap=0.05,
                template="plotly_dark",
                showlegend=False,
                height=350,
            )

            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

        st.subheader("Channels rated Medium or High")

        df_med_high = fdf[fdf["rating"].str.lower().isin(["medium", "high"])].copy()
        df_med_high = df_med_high.sort_values(
            ["rating", "mean_err"], ascending=[False, False]
        )

        rename_map = {
            "channel": "Channel",
            "rating": "Anomaly Level",
            "mean_err": "Mean RL",
            "std_err": "STD RL",
            "n_samples": "N"
        }


        cols_med_high = ["channel", "rating", "mean_err", "std_err", "n_samples"]
        df_display = df_med_high[cols_med_high].rename(columns=rename_map)

    
        def color_rating(val: str) -> str:
            v = str(val).lower()
            if v == "high":
                return "color: #ff4b4b; font-weight: 700;"
            if v == "medium":
                return "color: #ffb703; font-weight: 600;"
            return "color: #38b000;"

        styled = (
            df_display
            .style.format({
                "Mean RL": "{:.4f}",
                "STD RL": "{:.4f}"
            })
            .applymap(color_rating, subset=["Anomaly Level"])
        )

        st.dataframe(styled, use_container_width=True)




    with tab_pulse:
        st.subheader("Pulse-level anomaly details")

        channel = st.selectbox("Top 20 pulses with the highest reconstruction error for channel: ", options = selected_channels, index=None,
        placeholder="Select channel you would like to view...")
        st.write("You selected channel: ", channel)
        
        df_pulses_chan = fdf_pulses[fdf_pulses["channel"] == channel].copy()
        df_top20 = df_pulses_chan.nlargest(20, "RL")
        rename_map_pulse = {
            ("RL", "Reconstruction Loss"),
            ("EVT", "Event ID"),
        }

        features = ["channel", "Event ID", "Reconstruction Loss", "time"]

        pulse_display = df_top20.rename(columns=dict(rename_map_pulse))[features]
        st.dataframe(pulse_display, use_container_width = True)

        chan_dist = go.Figure()

        chan_dist.add_trace(
            go.Histogram(
                x=df_pulses_chan["RL"],
                nbinsx=50,
                marker=dict(
                    color="rgba(143, 72, 205, 0.75)",
                    line=dict(color="rgba(143, 72, 205, 1)", width=1.5)
                ),
                opacity=0.85,
                name="Reconstruction Loss",
            )
        )

        chan_dist.update_layout(
            title=f"Channel {channel} Reconstruction Loss Distribution",
            xaxis_title="Reconstruction Loss",
            yaxis_title="Pulse Frequency",
            bargap=0.05,
            template="plotly_dark",
            showlegend=False,
            height=350,
        )

        st.plotly_chart(chan_dist, use_container_width=True)
if __name__ == "__main__":
    main()
