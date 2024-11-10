import streamlit as st
import pandas as pd
import pydeck as pdk
from datetime import datetime, timedelta
from meteostat import Point, Hourly
from timezonefinder import TimezoneFinder
import numpy as np
import altair as alt
from garmin_fit_reader import GarminFitReader, SpeedUnit
import io
import math
import time

# Set page config
st.set_page_config(
    page_title="SailPerfView",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.session_data = None
if 'course_data' not in st.session_state:
    st.session_state.course_data = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'selected_metrics' not in st.session_state:
    st.session_state.selected_metrics = []
if 'available_metrics' not in st.session_state:
    st.session_state.available_metrics = []

# Convert wind speed from km/h to knots
@st.cache_data
def kmh_to_knots(kmh):
    return kmh * 0.539957

@st.cache_data
def get_timezone_from_coordinates(lat: float, lon: float) -> str:
    """
    Get the timezone string for given latitude and longitude coordinates.
    
    Args:
        lat (float): Latitude of the location
        lon (float): Longitude of the location
        
    Returns:
        str: Timezone string (e.g. 'America/New_York', 'Australia/Sydney')
    """
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=lat, lng=lon)
    
    if timezone_str is None:
        raise ValueError(f"Could not find timezone for coordinates: {lat}, {lon}")
        
    return timezone_str

@st.cache_data
def get_wind_data(lat, lon, start_time, end_time):
    """Fetch weather data from Meteostat"""
    try:
        sample_start = start_time - timedelta(hours=2)
        sample_end = end_time + timedelta(hours=2)
        timezone = get_timezone_from_coordinates(lat, lon)

        location = Point(lat, lon)
        df = Hourly(location, sample_start, sample_end, timezone)
        
        wind_df = df.fetch()[['wspd', 'wdir']]
                
        # Create a time series with hourly intervals
        time_range = pd.date_range(start=sample_start, periods=len(wind_df), freq='H')
        
        # Add the time series as a new column in the dataframe
        wind_df.insert(0, 'Time', time_range)

        # Convert Meteostat windpseed data from kph to knts
        wind_df['wspd'] = wind_df['wspd'].apply(kmh_to_knots)

        # Remove any rows where either wind speed or direction is null
        wind_df = wind_df.dropna(subset=['wspd', 'wdir'])

        return wind_df
    except Exception as e:
        st.warning(f"Unable to fetch wind data: {str(e)}")
        return None

def load_course_data(file):
    """Load course mark data from CSV file"""
    try:
        df = pd.read_csv(file)
        required_columns = ['Mark ID', 'Latitude', 'Longitude', 'Colour']

        if not all(col in df.columns for col in required_columns):
            st.error("Course file missing required columns")
            return None

        # Convert coordinates to float explicitly
        df['Latitude'] = df['Latitude'].astype(float)
        df['Longitude'] = df['Longitude'].astype(float)
        
        # Ensure color column is properly formatted
        df['Colour'] = df['Colour'].str.lower()
        
        return df
    except Exception as e:
        st.error(f"Error loading course data: {str(e)}")
        return None

def load_fit_file(file):
    """Load data from a FIT file"""
    try:
        # Save uploaded file to temporary file
        bytes_data = file.getvalue()
        temp_file = "temp.fit"
        with open(temp_file, "wb") as f:
            f.write(bytes_data)
        
        # Read FIT file
        fit_reader = GarminFitReader(temp_file, speed_unit=SpeedUnit.KNOTS)
        df = fit_reader.to_pandas()
        
        # Ensure required columns exist or create them
        if 'latitude_decimal' not in df.columns or 'longitude_decimal' not in df.columns:
            st.error("FIT file does not contain GPS data")
            return None
            
        # Rename columns to match expected format
        df = df.rename(columns={
            'timestamp': 'Time',
            'latitude_decimal': 'Latitude',
            'longitude_decimal': 'Longitude',
            f'speed_{SpeedUnit.KNOTS.value}': 'Speed',
            'altitude': 'Altitude'
        })
        
        # Add placeholder columns if they don't exist
        if 'LeanAngle' not in df.columns:
            df['LeanAngle'] = 0
        if 'Lap' not in df.columns:
            df['Lap'] = 1
        if 'GForceX' not in df.columns:
            df['GForceX'] = 0
        if 'GForceZ' not in df.columns:
            df['GForceZ'] = 0
        if 'GyroX' not in df.columns:
            df['GyroX'] = 0
        if 'GyroY' not in df.columns:
            df['GyroY'] = 0
        if 'GyroZ' not in df.columns:
            df['GyroZ'] = 0
        
        # Add Record column if not present
        if 'Record' not in df.columns:
            df['Record'] = range(len(df))
            
        return df
        
    except Exception as e:
        st.error(f"Error loading FIT file: {str(e)}")
        return None

@st.cache_data
def load_csv_data(file):
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(file)
        required_columns = [
            'Record', 'Time', 'Latitude', 'Longitude', 'Speed', 
            'LeanAngle', 'Altitude', 'GForceX', 'GForceZ', 'Lap',
            'GyroX', 'GyroY', 'GyroZ'
        ]
        
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file missing required columns")
            return None
            
        # Convert Time to datetime
        df['Time'] = pd.to_datetime(df['Time'])

        # Convert Speed from km/h to knots (1 km/h = 0.539957 knots)
        df['Speed'] = df['Speed'] * 0.539957
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV data: {str(e)}")
        return None

def get_plottable_columns(df):
    """Get columns that are numeric and make sense to plot"""
    exclude_columns = {'Record', 'Time', 'Latitude', 'Longitude', 'Lap'}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [col for col in numeric_cols if col not in exclude_columns]

def create_map_layer(df, current_index, highlight_index=None, course_data=None):
    """Create PyDeck map layers with highlight support"""
    layers = []

    # Create path data for the track
    path_data = pd.DataFrame({
        'path': [[[lon, lat] for lon, lat in zip(df['Longitude'], df['Latitude'])]],
    })

    # Path layer for the track
    path_layer = pdk.Layer(
        "PathLayer",
        data=path_data,
        get_path="path",
        width_scale=2,
        width_min_pixels=1,
        get_color=[0, 0, 255],
        pickable=True,
        auto_highlight=True
    )
    layers.append(path_layer)

    # Add course marks if available
    if course_data is not None and not course_data.empty:
        mark_points = []
        color_map = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'yellow': [255, 255, 0],
            'orange': [255, 165, 0],
            'white': [255, 255, 255],
            'black': [0, 0, 0]
        }
        
        for _, mark in course_data.iterrows():
            mark_points.append({
                'position': [float(mark['Longitude']), float(mark['Latitude'])],
                'color': color_map.get(str(mark['Colour']).lower(), [255, 255, 0]),
                'mark_id': str(mark['Mark ID'])
            })
        
        marks_df = pd.DataFrame(mark_points)
        
        marks_layer = pdk.Layer(
            "ColumnLayer",
            data=marks_df,
            get_position='position',
            get_fill_color='color',
            get_elevation=10,
            elevation_scale=5,
            radius=20,
            pickable=True,
            auto_highlight=True
        )
        layers.append(marks_layer)

    # Add current position marker
    current_pos = df.iloc[current_index:current_index+1]
    position_layer = pdk.Layer(
        "ScatterplotLayer",
        data=current_pos,
        get_position=["Longitude", "Latitude"],
        get_color=[255, 0, 0],
        get_radius=10,
        pickable=True
    )
    layers.append(position_layer)

    # Add highlight position if provided
    if highlight_index is not None:
        highlight_pos = df.iloc[highlight_index:highlight_index+1]
        highlight_layer = pdk.Layer(
            "ScatterplotLayer",
            data=highlight_pos,
            get_position=["Longitude", "Latitude"],
            get_color=[255, 255, 0],  # Yellow highlight
            get_radius=8,
            pickable=True
        )
        layers.append(highlight_layer)

    return layers

def create_deck(df, current_index, highlight_index=None, course_data=None):
    """Create PyDeck map view with highlight support"""
    center_lat = float(df['Latitude'].mean())
    center_lon = float(df['Longitude'].mean())
    
    zoom = 15

    layers = create_map_layer(df, current_index, highlight_index, course_data)
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        bearing=0
    )
    
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/satellite-v9',
        tooltip={
            "html": "<b>Mark ID:</b> {mark_id}",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    )
    
    return deck

def create_performance_plot(df, current_index, selected_metrics):
    """Create interactive performance visualization plot using Altair"""
    if not selected_metrics:
        return None
        
    plot_df = pd.DataFrame({
        'index': range(len(df)),
        'timestamp': df['Time'],
        **{metric: df[metric] for metric in selected_metrics}
    }).melt(id_vars=['index', 'timestamp'], var_name='Metric', value_name='Value')
    
    # Create selection for highlighting
    hover = alt.selection_point(
        fields=['index'],
        nearest=True,
        on='mouseover',
        empty=False,
    )
    
    # Create the base line chart
    lines = alt.Chart(plot_df).mark_line().encode(
        x=alt.X('index:Q', title='Time'),
        y=alt.Y('Value:Q', title='Value'),
        color='Metric:N'
    )
    
    # Add points for highlighting using add_params instead of depricated add_selection
    points = lines.mark_point().encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).add_params(hover)
 
    # Add vertical rule for current position
    rule = alt.Chart(pd.DataFrame({'x': [current_index]})).mark_rule(
        color='red',
        strokeWidth=2
    ).encode(x='x:Q')
    
    # Add tooltip
    tooltip = alt.Chart(plot_df).mark_rule().encode(
        x='index:Q',
        opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
        tooltip=[
            alt.Tooltip('timestamp:T', title='Time'),
            alt.Tooltip('Value:Q', title='Value'),
            alt.Tooltip('Metric:N', title='Metric')
        ]
    ).transform_filter(hover)
    
    chart = (lines + points + rule + tooltip).properties(
        height=400
    ).configure_axis(
        grid=True
    ).configure_view(
        strokeWidth=0
    )
    
    return chart


def main():
    st.title("⛵ Sailing Performance Viewer")
    
    # Initialize session states
    if 'map_key' not in st.session_state:
        st.session_state.map_key = 0
    if 'highlight_index' not in st.session_state:
        st.session_state.highlight_index = None
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'playing' not in st.session_state:
        st.session_state.playing = False
        
    # File upload section
    st.sidebar.header("Data Upload")
    file_type = st.sidebar.radio("Select file type:", ["CSV", "FIT"])
    uploaded_file = st.sidebar.file_uploader(f"Upload {file_type} file", type=[file_type.lower()])
    
    # Course file upload
    course_file = st.sidebar.file_uploader("Upload course marks (optional)", type=['csv'])
    
    # Handle course file upload
    if course_file is not None:
        course_data = load_course_data(course_file)
        if course_data is not None:
            st.session_state.course_data = course_data
            st.session_state.map_key += 1
    
    if uploaded_file is not None:
        # Load session data if not already loaded
        if st.session_state.session_data is None:
            if file_type == "CSV":
                df = load_csv_data(uploaded_file)
            else:  # FIT
                df = load_fit_file(uploaded_file)
                
            if df is not None:
                # Get wind data
                wind_data = get_wind_data(
                    df['Latitude'].mean(),
                    df['Longitude'].mean(),
                    df['Time'].min(),
                    df['Time'].max()
                )
                
                if wind_data is not None:
                    # Convert Time columns to datetime
                    df['Time'] = pd.to_datetime(df['Time'])
                    wind_data['Time'] = pd.to_datetime(wind_data['Time'])

                    # Merge with interpolation
                    df = pd.merge_asof(df.sort_values('Time'), 
                                     wind_data.sort_values('Time'), 
                                     on='Time', 
                                     direction='forward')

                    # Interpolate missing values
                    df[['wspd', 'wdir']] = df[['wspd', 'wdir']].interpolate()
                    df.rename(columns={'wspd': 'Wind speed (knts)', 
                                     'wdir': 'Wind direction (Deg)'}, 
                            inplace=True)

                st.session_state.session_data = df
                st.session_state.available_metrics = get_plottable_columns(df)
                default_metrics = ['Speed', 'Altitude']
                if 'Wind speed (knts)' in df.columns:
                    default_metrics.append('Wind speed (knts)')
                
                st.session_state.selected_metrics = default_metrics
                st.session_state.map_key += 1
        
        if st.session_state.session_data is not None:
            df = st.session_state.session_data
            
            # Metric selection
            st.sidebar.header("Plot Configuration")
            selected_metrics = st.sidebar.multiselect(
                "Select metrics to plot:",
                st.session_state.available_metrics,
                default=st.session_state.selected_metrics
            )
            st.session_state.selected_metrics = selected_metrics
            
            # Create containers
            map_container = st.empty()
            plot_container = st.empty()
            metrics_container = st.empty()
            timeline_container = st.container()
            
            # Timeline controls
            with timeline_container:
                col1, col2, col3, col4 = st.columns([1, 6, 1, 1])
                
                # Rewind button
                if col1.button("⏮"):
                    st.session_state.current_index = max(0, st.session_state.current_index - 10)
                    st.session_state.playing = False
                    
                # Timeline slider
                timeline_value = col2.slider(
                    "Timeline",
                    0,
                    len(df) - 1,
                    st.session_state.current_index,
                    key="timeline_slider"
                )
                
                # Forward button
                if col3.button("⏭"):
                    st.session_state.current_index = min(len(df) - 1, st.session_state.current_index + 10)
                    st.session_state.playing = False
                    
                # Play/Pause button
                play_button = col4.button("▶" if not st.session_state.playing else "⏸")
                if play_button:
                    st.session_state.playing = not st.session_state.playing

            # Update current index from slider
            st.session_state.current_index = timeline_value
            
            # Update visualization containers
            with map_container:
                deck = create_deck(
                    df, 
                    st.session_state.current_index,
                    st.session_state.highlight_index,
                    st.session_state.course_data
                )
                st.pydeck_chart(deck, use_container_width=True)
            
            with plot_container:
                if selected_metrics:
                    chart = create_performance_plot(df, st.session_state.current_index, selected_metrics)
                    if chart:
                        st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Please select at least one metric to plot")
            
            with metrics_container:
                st.subheader("Current Metrics")
                current_data = df.iloc[st.session_state.current_index]
                num_cols = min(4, len(selected_metrics))
                if num_cols > 0:
                    cols = st.columns(num_cols)
                    for i, metric in enumerate(selected_metrics[:num_cols]):
                        cols[i].metric(metric, f"{current_data[metric]:.1f}")
            
            # Handle playback
            if st.session_state.playing:
                if st.session_state.current_index < len(df) - 1:
                    st.session_state.current_index += 10
                    time.sleep(0.1)  # Adjust playback speed
                    st.rerun()
                else:
                    st.session_state.playing = False
                    st.rerun()

if __name__ == "__main__":
    main()