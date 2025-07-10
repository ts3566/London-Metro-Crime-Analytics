"""
Streamlit dashboard - Interactive visualization and prediction tool of burglaries in Greater London

================================================================
Run locally:
    streamlit run scripts/streamlit_app_12m.py

================================================================

"""
from __future__ import annotations
import pathlib, math, calendar
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import branca.colormap as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

# ── policing-resource constants ─────────────────────────────────────────────
OFFICERS_PER_WARD          = 100           # head-count per ward
SHIFTS_PER_OFFICER_WEEK    = 4             # <=4 burglary days each
HOURS_PER_SHIFT            = 2             # 2-h burglary window
SHIFTS_WEEKLY_CAP          = OFFICERS_PER_WARD * SHIFTS_PER_OFFICER_WEEK  # 400
CAPACITY_PER_SLOT          = OFFICERS_PER_WARD                            # <=100
SPECIAL_OP_PERIOD          = 4             # unchanged
# ── demand model parameters ────────────────────────────────────────────────
SHIFT_FACTOR_PER_BURG = 3.0           # extra shifts per predicted burglary
# one patrol pair (2 officers) in every one of the 56 weekly slots ⇒ 112 shifts
MIN_WEEKLY_SHIFTS     = 56 * 2        # 112

def weeks_in_month(ts: pd.Timestamp) -> float:          # new helper
    return calendar.monthrange(ts.year, ts.month)[1] / 7.0

# ── demand helper -----------------------------------------------------------
def weekly_shift_demand(predicted_burglaries: float) -> int:
    """Return integer officer-shifts needed per week for one ward."""
    demand  = max(MIN_WEEKLY_SHIFTS,
                MIN_WEEKLY_SHIFTS + math.ceil(predicted_burglaries * SHIFT_FACTOR_PER_BURG))
    demand  = min(demand, SHIFTS_WEEKLY_CAP)   # never exceed 400
    demand += demand % 2                       # force even (adds +1 if odd)
    return demand

# Uncapped version – used only for special-operation sizing
def raw_weekly_shift_demand(predicted_burglaries: float) -> int:
    demand  = max(MIN_WEEKLY_SHIFTS,
                  MIN_WEEKLY_SHIFTS + math.ceil(predicted_burglaries * SHIFT_FACTOR_PER_BURG))
    demand += demand % 2          # force even
    return demand                 # NO capacity clip

# ── time-slot helpers -------------------------------------------------------
TIME_SLOTS  = [(6,8),(8,10),(10,12),(12,14),(14,16),(16,18),(18,20),(20,22)]
SLOT_LABELS = [f"{s:02d}:00-{e:02d}:00" for s,e in TIME_SLOTS]

def sunset_hour(month:int) -> int:          # very coarse London sunset switch
    return 20 if 4 <= month <= 9 else 18    # Apr-Sep vs Oct-Mar

# ── WEEKLY schedule builder — every slot gets >= 1 pair (2 officers) ────────────
def build_weekly_schedule(shifts_week:int, month_ts:pd.Timestamp,
                          mult:float = 1.25,
                          baseline_pairs_per_slot:int = 1
                          ) -> tuple[pd.DataFrame, int]:
    """
    Returns a 7x8 table whose cells are 0,2,4… officers,
    **never less than two per slot**, and whose total equals `shifts_week`.
    """
    # safety: even & clipped
    shifts_week = min(shifts_week + shifts_week % 2, SHIFTS_WEEKLY_CAP)

    total_pairs    = shifts_week // 2
    base_pairs     = 56 * baseline_pairs_per_slot
    extra_pairs    = max(0, total_pairs - base_pairs)

    # slot weights (after-dark boost)
    sun_cut  = sunset_hour(month_ts.month)
    slot_w   = [mult if s >= sun_cut else 1.0 for s, _ in TIME_SLOTS]
    weights  = np.array(slot_w * 7, dtype=float)

    if extra_pairs:
        raw      = weights / weights.sum() * extra_pairs
        pairs    = np.floor(raw).astype(int)
        rem      = extra_pairs - pairs.sum()
        order    = np.argsort(raw - pairs)[::-1]
        pairs[order[:rem]] += 1              # distribute the remainder
    else:
        pairs    = np.zeros(56, dtype=int)

    pairs += baseline_pairs_per_slot         # add the chosen baseline (0 or 1)
    schedule  = (pairs * 2).reshape(7, len(TIME_SLOTS))

    df = pd.DataFrame(
        schedule,
        index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
        columns=SLOT_LABELS
    )
    return df, 0      # unmet=0 because demand <= 400 and baseline fits daily cap

# ── data paths --------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent  #go up one level to project root
DATA = ROOT / "data_cache" / "processed"
LOOK = ROOT / "data_cache" / "lookups"
PRED = ROOT / "predictions"

WARD_PANEL_FP = DATA / "ward_month_burglary.parquet"
LSOA_PANEL_FP = DATA / "lsoa_month_burglary.parquet"
WARD_GEO_JSON = LOOK / "wards_2024.geojson"
LSOA_GEO_JSON = LOOK / "LSOA21_Boundaries.geojson"
LOOKUP_CSV = LOOK / "LSOA21_WD24_Lookup.csv"
XGBOOST_PRED_CSV = PRED / "ward_burglary_predictions_12m.csv"

#configure Streamlit page settings
st.set_page_config(layout="wide")

# ── load --------------------------------------------------------------------
@st.cache_data
def load_ward_panel() -> pd.DataFrame:
    return pd.read_parquet(WARD_PANEL_FP)

@st.cache_data
def load_lsoa_panel() -> pd.DataFrame | None:
    try:
        return pd.read_parquet(LSOA_PANEL_FP)
    except FileNotFoundError:
        return None

@st.cache_data
def load_london_wards() -> set:
    lookup = pd.read_csv(LOOKUP_CSV)
    #filter for London borough codes (E09) but exclude City of London (E09000001)
    london_wards = lookup[
        (lookup['LAD24CD'].str.startswith('E09', na=False)) & 
        (lookup['LAD24CD'] != 'E09000001')
    ]['WD24CD'].unique()
    return set(london_wards)

@st.cache_data
def load_london_lsoas() -> set | None:
    lookup = pd.read_csv(LOOKUP_CSV)
    #filter for London borough codes (E09) but exclude City of London (E09000001)
    london_lsoas = lookup[
        (lookup['LAD24CD'].str.startswith('E09', na=False)) & 
        (lookup['LAD24CD'] != 'E09000001')
    ]['LSOA21CD'].unique()
    return set(london_lsoas)

@st.cache_resource
def load_ward_geo() -> gpd.GeoDataFrame:
    london_wards = load_london_wards()
    gdf = gpd.read_file(WARD_GEO_JSON)[["WD24CD", "WD24NM", "geometry"]]
    #filter for London wards
    gdf = gdf[gdf["WD24CD"].isin(london_wards)]
    gdf = gdf.to_crs(4326)  #lat/lon for web mapping
    return gdf

@st.cache_resource
def load_lsoa_geo() -> gpd.GeoDataFrame | None:
    try:
        london_lsoas = load_london_lsoas()
        gdf = gpd.read_file(LSOA_GEO_JSON)[["LSOA21CD", "LSOA21NM", "geometry"]]
        #filter for London LSOAs
        gdf = gdf[gdf["LSOA21CD"].isin(london_lsoas)]
        gdf = gdf.to_crs(4326)  #lat/lon for web mapping
        return gdf
    except FileNotFoundError:
        return None

@st.cache_data
def load_xgboost_predictions() -> pd.DataFrame:
    """Load XGBoost predictions for next 12 months."""
    df = pd.read_csv(XGBOOST_PRED_CSV, parse_dates=['Month'], infer_datetime_format=True)
    df["Predicted_Burglaries"] = pd.to_numeric(df["Predicted_Burglaries"], errors="coerce").fillna(0)
    return df

#load data
ward_panel = load_ward_panel()
lsoa_panel = load_lsoa_panel()
ward_geo = load_ward_geo()
lsoa_geo = load_lsoa_geo()
xgboost_pred = load_xgboost_predictions()

#get available months for both historical and next month forecast
historical_months = ward_panel["Month"].sort_values().unique()
forecast_month = xgboost_pred["Month"].min()  #use earliest forecast month

# ── sidebar controls --------------------------------------------------------
st.sidebar.title("London Burglary Dashboard")

# Analysis mode selection
view_mode = st.sidebar.radio("Type of Analysis",
    options=["Historical Data", "Future Forecast"],
    help="Historical Data shows actual burglary counts. Future Forecast shows predicted burglaries for upcoming months using XGBoost model."
)

#add forecast type selector when in Future Forecast mode
if view_mode == "Future Forecast":
	forecast_type = st.sidebar.radio("Forecast Type",
		options=["Ward Forecast", "LSOA Forecast"],
		help="Select forecast type for Future Forecast mode"
	)

#add forecast month selector if in forecast mode
selected_forecast_date = None
if view_mode == "Future Forecast":
    forecast_dates = sorted(
		(
			#use appropriate prediction dates based on forecast type.
			xgboost_pred["Month"].unique() 
			if forecast_type == "Ward Forecast" 
			else pd.read_csv(str(PRED / "lsoa_burglary_predictions_12m.csv"), parse_dates=['Month'])["Month"].unique()
		)
	)
    selected_forecast_date = st.sidebar.selectbox(
        "Select forecast month",
        options=sorted(forecast_dates),
        format_func=lambda x: x.strftime("%B %Y"),
        help="Select which month's forecast you want to view"
    )
    months_ahead = (selected_forecast_date.year - historical_months.max().year) * 12 + (selected_forecast_date.month - historical_months.max().month)
    
if view_mode == "Historical Data":
    view_level = st.sidebar.radio("View Level", 
        options=["Ward Level", "LSOA Level"],
        help="Ward Level shows data aggregated by electoral ward. LSOA (Lower Super Output Area) Level shows more detailed data at a smaller geographic level."
    )
#in forecast mode, set view_level based on forecast type
else:
    view_level = "Ward Level" if forecast_type == "Ward Forecast" else "LSOA Level"

if view_level == "LSOA Level" and (lsoa_panel is None or lsoa_geo is None):
    st.sidebar.error("LSOA level data is not available yet. Please use Ward Level view.")
    view_level = "Ward Level"

#use ward or LSOA data based on selection
panel = ward_panel if view_level == "Ward Level" else lsoa_panel
geo = ward_geo if view_level == "Ward Level" else lsoa_geo
id_col = "WD24CD" if view_level == "Ward Level" else "LSOA21CD"
name_col = "WD24NM" if view_level == "Ward Level" else "LSOA21NM"

#ensure panel is not None before proceeding
if panel is None:
    st.error("Selected data panel (Ward or LSOA) could not be loaded. Please check data availability.")
    st.stop()

def format_m(dt):
    return dt.strftime("%b %Y")

if view_mode == "Historical Data":
    sel_month = st.sidebar.selectbox("Select month", historical_months, format_func=format_m)
else:
    sel_month = forecast_month  #for forecast mode, we only have one month

#no longer showing individual burglary locations

# ── prepare data ------------------------------------------------------------
if view_mode == "Future Forecast":
    if forecast_type == "Ward Forecast":
        df_show = xgboost_pred[xgboost_pred['Month'] == selected_forecast_date].copy()
        
        #create mapping for wards
        ward_id_mapping = geo.set_index(name_col)[id_col].to_dict()
        ward_name_mapping = geo.set_index(id_col)[name_col].to_dict()
        
        #map ward IDs and names
        df_show[id_col] = df_show["Ward"].map(ward_id_mapping)
        df_show[name_col] = df_show["Ward"]  #keep the ward name for later use
        df_show["burglaries"] = df_show["Predicted_Burglaries"]
    else:
        #LSOA Forecast: load LSOA predictions from file
        lsoa_pred_fp = PRED / "lsoa_burglary_predictions_12m.csv"
        df_lsoa_pred = pd.read_csv(lsoa_pred_fp, parse_dates=['Month'])
        df_show = df_lsoa_pred[df_lsoa_pred['Month'] == selected_forecast_date].copy()
        # Rename column so that it matches the geo merge key
        df_show = df_show.rename(columns={"LSOA": "LSOA21CD"})
        #create mapping for LSOAs from geo file
        lsoa_name_mapping = geo.set_index("LSOA21CD")["LSOA21NM"].to_dict()
        df_show["LSOA21NM"] = df_show["LSOA21CD"].map(lsoa_name_mapping)
        #remove rows with missing LSOA names
        df_show = df_show[df_show["LSOA21NM"].notna()]
        df_show["burglaries"] = df_show["Predicted_Burglaries"]
else:
    df_show = panel[panel["Month"] == sel_month].copy()

#merge geometry and handle missing values
chor = geo.merge(df_show[[id_col, "burglaries"]], left_on=id_col, right_on=id_col, how="left")
chor.loc[:, "burglaries"] = chor["burglaries"].fillna(0)

#normalize burglary counts for colormap
norm = colors.Normalize(vmin=chor['burglaries'].min(), vmax=chor['burglaries'].max())
colormap = plt.colormaps.get_cmap('Reds')  # Changed from cm.get_cmap
chor['fill_color'] = chor['burglaries'].apply(lambda x: colormap(norm(x))[:3])
#apply opacity: 0.75
chor['fill_color'] = chor['fill_color'].apply(lambda rgb: [int(c * 255) for c in rgb] + [0.75*255])

# ── main layout -------------------------------------------------------------
#update title and subtitle
st.title("Interactive visualization and prediction tool of burglaries in Greater London")
if view_mode == "Future Forecast":
    subtitle = f"Forecast for {selected_forecast_date.strftime('%B %Y')}"
else:
    subtitle = f"{sel_month.strftime('%B %Y')}"
st.markdown(f"## {subtitle}")

#calculate center coordinates of London
bounds = chor.total_bounds  #returns (min x, min y, max x , max y)
mid_lon = (bounds[0] + bounds[2]) / 2
mid_lat = (bounds[1] + bounds[3]) / 2

#create Folium map - dynamic sizing with dark mode
m = folium.Map(
    location=[mid_lat, mid_lon],
    zoom_start=10,
    tiles='cartodbpositron'
)

#create colormap
min_value = chor['burglaries'].min()
max_value = chor['burglaries'].max()
colormap = cm.LinearColormap(
    colors=['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
    vmin=min_value,
    vmax=max_value,
    caption='Number of Burglaries'
)
colormap.add_to(m)

#add choropleth layer
folium.GeoJson(
    chor.to_json(),
    name='Burglaries',
    style_function=lambda x: {
        'fillColor': colormap(x['properties']['burglaries']),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7
    },
    tooltip=folium.GeoJsonTooltip(
        fields=[name_col, 'burglaries'],
        aliases=[name_col.replace('NM', ' Name'), 'Burglaries'],
        style=('background-color: steelblue; color: white; font-family: arial; font-size: 12px; padding: 10px;')
    )
).add_to(m)

#add layer control
folium.LayerControl().add_to(m)

#create a container for the map with dynamic sizing
map_container = st.container()
with map_container:
    #custom CSS to make the map container responsive and full-width
    st.markdown(
        """
        <style>
        [data-testid="stHorizontalBlock"] > div {
            width: 100%;
        }
        .st-emotion-cache-16txtl3  {
            padding: 1rem;
        }
        iframe {
            width: 100% !important;
            min-height: 800px !important;
            border: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    #display map using st_folium with explicit height
    st_folium(m, width='100%', height=800)

#display wards with min and max burglary counts
min_burglaries_value = chor['burglaries'].min()
max_burglaries_value = chor['burglaries'].max()

min_burglary_areas = chor[chor['burglaries'] == min_burglaries_value]
max_burglary_areas = chor[chor['burglaries'] == max_burglaries_value]

area_type_display = "LSOAs" if name_col == "LSOA21NM" else "Wards"

min_area_names_list = min_burglary_areas[name_col].unique()
if len(min_area_names_list) > 5:
    min_area_names = ", ".join(min_area_names_list[:5]) + f", and {len(min_area_names_list) - 5} more"
else:
    min_area_names = ", ".join(min_area_names_list)

max_area_names_list = max_burglary_areas[name_col].unique()
if len(max_area_names_list) > 5:
    max_area_names = ", ".join(max_area_names_list[:5]) + f", and {len(max_area_names_list) - 5} more"
else:
    max_area_names = ", ".join(max_area_names_list)

st.markdown(f"{area_type_display} with Minimum Burglaries ({min_burglaries_value}): {min_area_names}")
st.markdown(f"{area_type_display} with Maximum Burglaries ({max_burglaries_value}): {max_area_names}")

#historical data section: Replace expander for data table with inline display
if view_mode == "Historical Data":
    search = st.text_input(
        "Search by " + ("Ward name or code" if view_level == "Ward Level" else "LSOA name or code"),
        key="historical_search"
    )
    df_table = chor[[id_col, name_col, "burglaries"]].copy()
    df_table.columns = ["Code", "Name", "Burglaries"]
    if search:
        mask = (df_table["Name"].str.contains(search, case=False)) | (df_table["Code"].str.contains(search, case=False))
        df_table = df_table[mask]
    st.dataframe(df_table.sort_values("Burglaries", ascending=False), use_container_width=True)

# ── Display forecast table and plots -------------------------------------------------------------
if view_mode == "Future Forecast":
    if forecast_type == "Ward Forecast":
        search = st.text_input("Search by Ward name or code", "")
        combined_df = df_show[[id_col, name_col, "Predicted_Burglaries"]].copy()
        combined_df.columns = ["Code", "Name", "Predicted Burglaries"]
        if search:
            mask = (combined_df["Name"].str.contains(search, case=False)) | (combined_df["Code"].str.contains(search, case=False))
            combined_df = combined_df[mask]
        st.dataframe(combined_df.sort_values("Predicted Burglaries", ascending=False), use_container_width=True)
    else:  #LSOA Forecast table
        search = st.text_input("Search by LSOA name or code", "")
        combined_df = df_show[["LSOA21CD", "LSOA21NM", "Predicted_Burglaries"]].copy()
        combined_df.columns = ["Code", "Name", "Predicted Burglaries"]
        if search:
            mask = (combined_df["Name"].str.contains(search, case=False)) | (combined_df["Code"].str.contains(search, case=False))
            combined_df = combined_df[mask]
        st.dataframe(combined_df.sort_values("Predicted Burglaries", ascending=False), use_container_width=True)
        
    if forecast_type == "Ward Forecast":
        available_wards = sorted(df_show["Ward"].unique())
        selected_ward_filter = st.selectbox("Select Ward for Plot", options=available_wards, index=0)
        
        import altair as alt
        
        st.markdown("#### Historical vs Forecast for " + selected_ward_filter)
        hist_data = panel[(panel["WD24NM"] == selected_ward_filter) &
                          (panel["Month"].dt.month == selected_forecast_date.month)].copy()
        if hist_data.empty:
            st.write("No historical data available for the selected ward and month.")
        else:
            hist_data["Year_Month"] = hist_data["Month"].dt.strftime("%b %Y")
            hist_chart = alt.Chart(hist_data).mark_line(point=True).transform_calculate(
                Type="'Historical Trend'"
            ).encode(
                x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("burglaries:Q", title="Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Historical Trend", "Forecast Trend"],
                    range=["#ADD8E6", "#FF4B4B"]
                ), legend=alt.Legend(title="Trend"))
            ).properties(
                title=f"Historical Burglaries for {selected_ward_filter} in {selected_forecast_date.strftime('%B')} over the Years",
                height=400
            )
            forecast_subset = df_show[df_show["Ward"] == selected_ward_filter]
            if forecast_subset.empty:
                st.write("No forecast data available for the selected ward.")
            else:
                forecast_value = forecast_subset["Predicted_Burglaries"].iloc[0]
                forecast_month_str = selected_forecast_date.strftime("%b %Y")
                forecast_df = pd.DataFrame({"Year_Month": [forecast_month_str], "Forecast": [forecast_value]})
                forecast_point = alt.Chart(forecast_df).mark_point(color="#FF4B4B", size=100).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Forecast:Q", title="Burglaries")
                )
                last_hist_value = hist_data.sort_values("Month")["burglaries"].iloc[-1]
                last_hist_label = hist_data.sort_values("Month")["Year_Month"].iloc[-1]
                connect_df = pd.DataFrame({
                    "Year_Month": [last_hist_label, forecast_month_str],
                    "Burglaries": [last_hist_value, forecast_value]
                })
                connect_line = alt.Chart(connect_df).mark_line(strokeDash=[5,3]).transform_calculate(
                    Type="'Forecast Trend'"
                ).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Burglaries:Q", title="Burglaries"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["Historical Trend", "Forecast Trend"],
                        range=["#ADD8E6", "#FF4B4B"]
                    ), legend=alt.Legend(title="Trend"))
                )
                combined_chart = hist_chart + forecast_point + connect_line
                st.altair_chart(combined_chart, use_container_width=True)

        st.markdown(" ")        
        st.markdown("#### Forecast Trend over Next 12 Months")
        trend_data = xgboost_pred[xgboost_pred["Ward"] == selected_ward_filter].copy()
        if trend_data.empty:
            st.write("No forecast trend data available for the selected ward.")
        else:
            trend_data.sort_values("Month", inplace=True)
            trend_chart = alt.Chart(trend_data).mark_line(point=True, strokeDash=[5,3]).transform_calculate(
                Type="'Forecast Trend'"
            ).encode(
                x=alt.X("Month:T", axis=alt.Axis(format="%b %Y", title="Month and Year", labelAngle=0, tickCount=12)),
                y=alt.Y("Predicted_Burglaries:Q", title="Predicted Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Forecast Trend"], range=["#FF4B4B"]), legend=alt.Legend(title="Trend"))
            ).properties(
                title="Forecast Trend over Next 12 Months for " + selected_ward_filter,
                height=400
            )
            st.altair_chart(trend_chart, use_container_width=True)
    else:  #LSOA Forecast plots
        available_lsoas = sorted(df_show["LSOA21NM"].dropna().unique())
        selected_lsoa_filter = st.selectbox("Select LSOA for Plot", options=available_lsoas, index=0)
        
        import altair as alt
        
        st.markdown("#### Historical vs Forecast for " + selected_lsoa_filter)
        hist_data = panel[(panel["LSOA21NM"] == selected_lsoa_filter) &
                          (panel["Month"].dt.month == selected_forecast_date.month)].copy()
        if hist_data.empty:
            st.write("No historical data available for the selected LSOA and month.")
        else:
            hist_data["Year_Month"] = hist_data["Month"].dt.strftime("%b %Y")
            hist_chart = alt.Chart(hist_data).mark_line(point=True).transform_calculate(
                Type="'Historical Trend'"
            ).encode(
                x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("burglaries:Q", title="Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["Historical Trend", "Forecast Trend"],
                    range=["#ADD8E6", "#FF4B4B"]
                ), legend=alt.Legend(title="Trend"))
            ).properties(
                title=f"Historical Burglaries for {selected_lsoa_filter} in {selected_forecast_date.strftime('%B')} over the Years",
                height=400
            )
            forecast_subset = df_show[df_show["LSOA21NM"] == selected_lsoa_filter]
            if forecast_subset.empty:
                st.write("No forecast data available for the selected LSOA.")
            else:
                forecast_value = forecast_subset["Predicted_Burglaries"].iloc[0]
                forecast_month_str = selected_forecast_date.strftime("%b %Y")
                forecast_df = pd.DataFrame({"Year_Month": [forecast_month_str], "Forecast": [forecast_value]})
                forecast_point = alt.Chart(forecast_df).mark_point(color="#FF4B4B", size=100).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Forecast:Q", title="Burglaries")
                )
                last_hist_value = hist_data.sort_values("Month")["burglaries"].iloc[-1]
                last_hist_label = hist_data.sort_values("Month")["Year_Month"].iloc[-1]
                connect_df = pd.DataFrame({
                    "Year_Month": [last_hist_label, forecast_month_str],
                    "Burglaries": [last_hist_value, forecast_value]
                })
                connect_line = alt.Chart(connect_df).mark_line(strokeDash=[5,3]).transform_calculate(
                    Type="'Forecast Trend'"
                ).encode(
                    x=alt.X("Year_Month:O", title="Month and Year", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Burglaries:Q", title="Burglaries"),
                    color=alt.Color("Type:N", scale=alt.Scale(
                        domain=["Historical Trend", "Forecast Trend"],
                        range=["#ADD8E6", "#FF4B4B"]
                    ), legend=alt.Legend(title="Trend"))
                )
                combined_chart = hist_chart + forecast_point + connect_line
                st.altair_chart(combined_chart, use_container_width=True)

        st.markdown(" ")        
        st.markdown("#### Forecast Trend over Next 12 Months")
        import altair as alt
        trend_data = pd.read_csv(str(PRED / "lsoa_burglary_predictions_12m.csv"), parse_dates=['Month'])
        #get the selected LSOA code based on the chosen LSOA name from the dropdown
        selected_lsoa_code = df_show.loc[df_show["LSOA21NM"] == selected_lsoa_filter, "LSOA21CD"].iloc[0]
        trend_data = trend_data[trend_data["LSOA"] == selected_lsoa_code].copy()
        if trend_data.empty:
            st.write("No forecast trend data available for the selected LSOA.")
        else:
            trend_data.sort_values("Month", inplace=True)
            trend_chart = alt.Chart(trend_data).mark_line(point=True, strokeDash=[5,3]).transform_calculate(
                Type="'Forecast Trend'"
            ).encode(
                x=alt.X("Month:T", axis=alt.Axis(format="%b %Y", title="Month and Year", labelAngle=0, tickCount=12)),
                y=alt.Y("Predicted_Burglaries:Q", title="Predicted Burglaries"),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Forecast Trend"], range=["#FF4B4B"]), legend=alt.Legend(title="Trend"))
            ).properties(
                title="Forecast Trend over Next 12 Months for " + selected_lsoa_filter,
                height=400
            )
            st.altair_chart(trend_chart, use_container_width=True)

if view_mode == "Future Forecast":
    show_alloc = st.sidebar.checkbox(
        "Police resource allocation", value=True,
        help="Toggle recommended officer assignments based on predicted "
             "burglary demand and capacity/stress scores."
    )

    after_dark_mult = st.sidebar.slider(
    "After-dark multiplier (18:00/20:00-22:00)",
    1.0, 4.0, 2.0, 0.1,
    help="Weight evening slots more heavily when distributing the weekly officer shifts."
    )

else:
    show_alloc = False

if show_alloc and view_mode == "Future Forecast":

    if forecast_type == "Ward Forecast":

        st.markdown("### Officer allocation per ward (2-hour burglary window)")

        df_alloc = df_show.copy()

        #weekly shift demand & stress
        df_alloc["Allocated officers/week"] = df_alloc["Predicted_Burglaries"].apply(weekly_shift_demand)
        df_alloc["Stress"]        = df_alloc["Allocated officers/week"] / SHIFTS_WEEKLY_CAP

        alloc_tbl = (
            df_alloc[[id_col, name_col, "Predicted_Burglaries",
                    "Allocated officers/week", "Stress"]]
            .rename(columns={id_col: "Code",
                            name_col: "Ward",
                            "Predicted_Burglaries": "Pred burglaries"})
            .sort_values("Stress", ascending=False)
        )

        #add search input for filtering by Ward code or name
        search_alloc = st.text_input("Search by Ward code or name", key="alloc_search")
        if search_alloc:
            alloc_tbl = alloc_tbl[
                alloc_tbl["Code"].str.contains(search_alloc, case=False) |
                alloc_tbl["Ward"].str.contains(search_alloc, case=False)
            ]
        st.dataframe(alloc_tbl, use_container_width=True)

        # ── NEW: day-by-day 2-h slot schedule ─────────────────────────────────
        st.markdown("#### Weekly schedule for selected ward (number of allocated officers per 2-hour slot)")

        ward_choice = st.selectbox("Select ward for schedule",
                                options=df_alloc[name_col].values,
                                index=0, key="sched_ward")

        pred_burg    = df_alloc.loc[df_alloc[name_col]==ward_choice,
                                    "Predicted_Burglaries"].iloc[0]
        shifts_week = weekly_shift_demand(pred_burg)

        sched_df, unmet = build_weekly_schedule(shifts_week,
                                                selected_forecast_date,
                                                after_dark_mult,
                                                baseline_pairs_per_slot=1)

        st.dataframe(sched_df, use_container_width=True)

        if unmet > 0:
            st.info(f"Weekly capacity saturated - **{unmet} officer-shifts** could "
                    "not be scheduled within the 4-day-per-officer rule. "
                    "Consider overtime or a special operation.")

        # ── Special-operation targets (once / 4 months) ───────────────────────────
        next4     = pd.date_range(selected_forecast_date, periods=4, freq="MS")
        future4   = xgboost_pred[xgboost_pred["Month"].isin(next4)].copy()

        # raw (uncapped) weekly demand
        future4["raw_week_shifts"] = future4["Predicted_Burglaries"].apply(raw_weekly_shift_demand)

        # overflow above routine capacity
        future4["over_cap"] = future4["raw_week_shifts"] - SHIFTS_WEEKLY_CAP
        future4["over_cap"] = future4["over_cap"].clip(lower=0)

        # buffer when stress >= 0.80  (adds 20 % extra, rounded down to even)
        future4["stress"]   = future4["raw_week_shifts"] / SHIFTS_WEEKLY_CAP
        future4["buffer"]   = np.where(
            future4["stress"] >= 0.80, (0.20 * future4["raw_week_shifts"] // 2 * 2).astype(int), 0)

        # total extra shifts required for a special-op week
        future4["extra"] = (future4["over_cap"] + future4["buffer"]).astype(int)
        future4["extra"] = (future4["extra"] // 2 * 2)   # ensure even

        # summarise: one row per ward over the 4-month window
        spec_tbl = (future4.groupby("Ward")[["extra"]]
                    .max()                        # max requirement in the window
                    .query("extra > 0")
                    .sort_values("extra", ascending=False)
                    .head(10)
                    .rename(columns={"extra": "Extra officers/week"}))

        st.markdown("#### Suggested special-operation wards (next 4 months)")
        if spec_tbl.empty:
            st.write("No ward exceeds the surge-trigger threshold in the next four-month window.")
        else:
            st.table(spec_tbl)
            
        st.markdown(
            "Note: A ward is considered 'stressed' when its raw weekly shift demand reaches at least 80% of the total capacity. "
            "For such wards, an extra buffer equivalent to 20% of the raw demand (rounded down to an even number) is added to account for potential surges."
        )

    else:  #LSOA Forecast

        st.markdown("### Within-ward shift split across LSOAs")

        lkp = pd.read_csv(LOOKUP_CSV)[["LSOA21CD", "WD24CD", "WD24NM"]]
        lsoa_w = df_show.merge(lkp, on="LSOA21CD", how="left")

        wards_available = sorted(lsoa_w["WD24NM"].dropna().unique())
        sel_ward = st.selectbox("Ward to split its 700 officers/week (100 officers/day)", wards_available)

        #compute aggregated ward-level predictions to determine total predicted burglaries and required officers per ward
        ward_alloc = lsoa_w.groupby("WD24NM").agg({"Predicted_Burglaries": "sum"}).reset_index()
        ward_alloc["Allocated officers/week"] = ward_alloc["Predicted_Burglaries"].apply(weekly_shift_demand)
        ward_need = int(ward_alloc.loc[ward_alloc["WD24NM"] == sel_ward, "Allocated officers/week"].iloc[0])

        total_burg = lsoa_w[lsoa_w["WD24NM"] == sel_ward]["Predicted_Burglaries"].sum()

        if ward_need == 0 or total_burg == 0:
            st.info("No officers required for this ward.")
        else:
            # ── proportional split of `ward_need` shifts across LSOAs (even numbers) ──
            ward_lsoas = lsoa_w[lsoa_w["WD24NM"] == sel_ward].copy()

            # 1. raw proportional share
            ward_lsoas["raw"] = (ward_lsoas["Predicted_Burglaries"] /
                                total_burg * ward_need)

            # 2. round **down** to nearest even number (pairs)
            ward_lsoas["even"] = (ward_lsoas["raw"] // 2 * 2).astype(int)

            # 3. distribute leftover pairs to highest fractional parts
            pairs_left   = (ward_need - ward_lsoas["even"].sum()) // 2
            ward_lsoas["fraction"] = ward_lsoas["raw"] - ward_lsoas["even"]
            ward_lsoas.sort_values("fraction", ascending=False, inplace=True)
            ward_lsoas.iloc[:pairs_left, ward_lsoas.columns.get_loc("even")] += 2

            # 4. tidy up
            ward_lsoas.sort_values("even", ascending=False, inplace=True)
            ward_lsoas.rename(columns={"even": "Allocated officers/week"}, inplace=True)
            lsoa_alloc_tbl = ward_lsoas[["LSOA21CD", "LSOA21NM",
                                        "Predicted_Burglaries", "Allocated officers/week"]].rename(
                columns={"LSOA21CD": "LSOA code",
                        "LSOA21NM": "LSOA",
                        "Predicted_Burglaries": "Pred burglaries"}
            )

            #new: add search input for filtering by LSOA Code or Name
            search_alloc_lsoa = st.text_input("Search by LSOA code or name", key="lsoa_alloc_search")
            if search_alloc_lsoa:
                lsoa_alloc_tbl = lsoa_alloc_tbl[
                    lsoa_alloc_tbl["LSOA code"].str.contains(search_alloc_lsoa, case=False) |
                    lsoa_alloc_tbl["LSOA"].str.contains(search_alloc_lsoa, case=False)
                ]
            st.dataframe(lsoa_alloc_tbl, use_container_width=True)

            # ── LSOA-level schedule ---------------------------------------------------------
            st.markdown("#### Weekly schedule for selected LSOA (number of allocated officers per 2-hour slot)")

            lsoa_choice = st.selectbox("Select LSOA for schedule",
                                    options=lsoa_alloc_tbl["LSOA"].values,
                                    key="lsoa_schedule_choice")

            lsoa_shifts = int(lsoa_alloc_tbl.loc[
                lsoa_alloc_tbl["LSOA"] == lsoa_choice, "Allocated officers/week"
            ].iloc[0])

            lsoa_sched_df, _ = build_weekly_schedule(
                lsoa_shifts,
                selected_forecast_date,
                after_dark_mult,
                baseline_pairs_per_slot=0          # no automatic pair per slot
            )

            st.dataframe(lsoa_sched_df, use_container_width=True)
            