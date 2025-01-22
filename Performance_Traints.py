import streamlit as st
import time
import numpy as np
import pandas as pd
import time
import plotly.express as px


def insightConstructor():
    import streamlit as st
    import Sensors as sensor
    import time
    import numpy as np
    import pandas as pd
    import time
    import plotly.express as px
    from streamlit_option_menu import option_menu

    selected_farm = st.sidebar.selectbox("Select the Farm:", ["UCTC", "INTROP"])
    selected_farm = st.sidebar.selectbox("Select the Plant:", ["Pak Choy-Side1", "Pak Choy-Side2"])

    def printCostumTitleAndContenth3(title, context):
        return f"""
            <div class="jumbotron">
            <h3>{title}</h3>
            <h6>{context}</h6>
            </div>
            <div class="container">
            </div>
            """

    def printCostumTitleAndContenth2(title, context):
        return f"""
            <div class="jumbotron">
            <h2>{title}</h2>
            <h6>{context}</h6>
            </div>
            <div class="container">
            </div>
            """

    def printCostumTitleAndContenth1(title, context):
        return f"""
            <div class="jumbotron">
            <h1>{title}</h1>
            <h5>{context}</h5>
            </div>
            <div class="container">
            </div>
            """

    def printCostumTitleAndContenth4(title, context):
        return f"""
            <div class="jumbotron">
            <h4>{title}</h4>
            <h4>{context}</h4>
            </div>
            <div class="container">
            </div>
            """

    def animated_linear_progress_bar(label, value, color='green'):
        progress_html = f"""
            <svg width="300" height="30" style="background-color: #f1f1f1; border-radius: 5px;">
                <rect id="progress-rect" width="0%" height="100%" fill="{color}">
                    <animate attributeName="width" from="0%" to="{value}%" dur="2s" fill="freeze" />
                </rect>
                <text x="50%" y="50%" fill="black" font-size="14" font-weight="bold" text-anchor="middle" dy=".3em">{label}</text>
            </svg>

            <script>
                const progressRect = document.getElementById('progress-rect');
                progressRect.setAttribute('width', '{value}%');
            </script>
        """
        st.markdown(progress_html, unsafe_allow_html=True)

    # Example usage with animated linear progress bar

    def animated_circular_progress_bar(label, value, max_value, color='red', max_size=150):
        normalized_value = min(value / max_value, 1.0)  # Normalize value to be between 0 and 1
        progress_html = f"""
            <div id="progress-container" style="width: {max_size}px; height: {max_size}px; position: relative; border-radius: 50%; overflow: hidden;">
                <div id="progress-circle" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></div>
                <div id="animated-circle" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: black ; font-size: 11px; font-weight: bold;">{label}<br>{value} </div>
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/progressbar.js/1.0.1/progressbar.min.js"></script>
            <script>
                const container = document.getElementById('progress-container');
                const bar = new ProgressBar.Circle(container, {{
                    strokeWidth: 13,
                    easing: 'easeInOut',
                    duration: 2000,
                    color: '{color}',
                    trailColor: '#e0e0e0',
                    trailWidth: 10,
                    svgStyle: null
                }});

                bar.animate({normalized_value});
            </script>
        """
        return progress_html

    def animated_linear_progress_bar_with_metric(metric_value, label, value, color='green', width=200, height=20):
        progress_html = f"""
            <div style="display: flex; align-items: center; text-align: left;">
                <div style="font-size: 14px; font-weight: bold; margin-right: 10px;">{metric_value}</div>
                <div style="position: relative; width: {width}px;">
                    <svg width="{width}" height="{height}" style="background-color: #f1f1f1; border-radius: 5px;">
                        <rect id="progress-rect" x="0" y="0" width="0%" height="100%" fill="{color}">
                            <animate attributeName="width" from="0%" to="{value}%" dur="2s" fill="freeze" />
                        </rect>
                        <text x="50%" y="50%" fill="black" font-size="14" font-weight="bold" text-anchor="middle" dy=".3em">{label}</text>
                    </svg>
                </div>
            </div>

            <script>
                const progressRect = document.getElementById('progress-rect');
                progressRect.setAttribute('width', '{value}%');
            </script>
        """
        st.markdown(progress_html, unsafe_allow_html=True)

    # HTML and CSS for animated line
    animated_line_html = """
    <style>
        @keyframes drawLine {
            to {
                stroke-dashoffset: 0;
            }
        }

        .animated-line {
            width: 100%;
            height: 12px;
            background-color: black;
            position: relative;
            overflow: hidden;
        }

        .line-path {
            stroke-dasharray: 1000;
            stroke-dashoffset: 1000;
            animation: drawLine 2s forwards;
            stroke: #3498db;
            stroke-width: 2px;
        }
    </style>

    <div class="animated-line">
        <svg width="100%" height="100%">
            <line class="line-path" x1="0" y1="1" x2="100%" y2="1"/>
        </svg>
    </div>
    """

    def color_cell(value, best_value, lower_limit, upper_limit):
        if value == best_value:
            return f'<span style="background-color: green; padding: 10px; display: block; font-weight: bold;">{value}</span>'
        elif value < best_value * 0.15:
            return f'<span style="background-color:red; padding: 10px; display: block; font-weight: bold;">{value}</span>'
        elif value < best_value * 0.45:
            return f'<span style="background-color: #ff6666; padding: 10px; display: block; font-weight: bold;">{value}</span>'
        elif value < best_value * 0.85:
            return f'<span style="background-color: #ffcc99; padding: 10px; display: block; font-weight: bold;">{value}</span>'
        elif value > best_value * 0.85:
            return f'<span style="background-color: #b3ffb3; padding: 10px; display: block; font-weight: bold;">{value}</span>'

    def color_cell2(best_value, value, lower_limit, upper_limit):
        if value == best_value:
            return f'green'
        elif value < best_value * 0.15:
            return f'red'
        elif value < best_value * 0.45:
            return f'orange'
        elif value < best_value * 0.85:
            return f'yellow'
        else:
            return 'green'


    # Data
    option2 = "Pak choy"


    if option2 == 'Pak choy':
        import streamlit as st
        import pandas as pd
        import plotly.graph_objects as go
        import datetime
        st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Explore Yield, Sensor Values and Traits Over Time</h2>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date(2025, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.date(2025, 1, 15))

        # Example Data: Yield data for 4 cycles
        data = {
            'Cycle': ['Cycle 1', 'Cycle 2', 'Cycle 3', 'Cycle 4'],
            'Actual': [45, 50, 55, 60],
            'Target': [50, 55, 60, 65],
            'Predicted': [47, 52, 58, 63]
        }

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Create a figure using Plotly Graph Objects
        fig = go.Figure()
        with st.expander('Yeild'):
            # Add bars for Actual and Target
            fig.add_trace(go.Bar(
                x=df['Cycle'],
                y=df['Actual'],
                name='Actual',
                marker_color='blue'
            ))

            fig.add_trace(go.Bar(
                x=df['Cycle'],
                y=df['Target'],
                name='Target',
                marker_color='green'
            ))

            # Add dots and lines for Predicted
            fig.add_trace(go.Scatter(
                x=df['Cycle'],
                y=df['Predicted'],
                mode='lines+markers',
                name='Predicted',
                marker=dict(color='red', size=10),
                line=dict(color='red', width=2, dash='dot')
            ))

            # Update layout for better visualization
            fig.update_layout(
                title='Yield Comparison for 4 Cycles',
                xaxis_title='Cycle',
                yaxis_title='Yield',
                barmode='group',
                template='plotly_white',
                legend_title='Yield Type',
                xaxis=dict(tickmode='linear'),
                yaxis=dict(gridcolor='lightgrey'),
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            import pandas as pd
            import random

            # Function to generate color based on value
            def color_cell(value, max_value, min_value):
                normalized = (value - min_value) / (max_value - min_value + 1e-5)
                if normalized > 0.7:
                    return "green"
                elif normalized > 0.4:
                    return "orange"
                else:
                    return "red"

            # Function to color Pot Number (example logic)
            def pot_number_color(value):
                if value % 2 == 0:  # Even pots
                    return "lightblue"
                else:  # Odd pots
                    return "lightgreen"

            # Generate data for 4 cycles and 2 pots per cycle
            import pandas as pd

            # Given information
            data = {
                'Cycle': ['Cycle 1', 'Cycle 2', 'Cycle 3', 'Cycle 4'],
                'Actual Yield': [45, 50, 55, 60],
                'Target': [50, 55, 60, 65],
                'Predicted': [47, 52, 58, 63]
            }

            # Create a DataFrame
            df = pd.DataFrame(data)

            # Expand the dataset to include 2 pots per cycle
            expanded_data = []
            for index, row in df.iterrows():
                for pot in range(1, 3):  # 2 pots per cycle
                    expanded_data.append({
                        "Cycle": row["Cycle"],
                        "Pot Number": pot,
                        "Actual Yield": row["Actual Yield"],
                        "Target": row["Actual Yield"] + 15,
                        "Predicted": row["Predicted"]
                    })

            # Create the expanded DataFrame
            expanded_df = pd.DataFrame(expanded_data)

            # Calculate additional correlated values
            expanded_df['AVG Plant Height'] = expanded_df['Actual Yield'] * 0.8  # Example correlation factor
            expanded_df['AVG Longest Leaf'] = expanded_df['Actual Yield'] * 0.6  # Example correlation factor
            expanded_df['AVG Leaves Count'] = expanded_df['Actual Yield'] * 0.15  # Example correlation factor



            # Now, expanded_df is ready to be used for any further processing or visualization in your Streamlit app.

            # Create DataFrame
            df = expanded_df

            # HTML Table Construction
            html_table = """
            <div style="overflow-x:auto;">
                <table style="width: 100%; border-collapse: collapse; text-align: center; font-family: Arial, sans-serif;">
                    <thead>
                        <tr style="background-color: #f2f2f2; font-weight: bold;">
                            <th style="border: 1px solid black; padding: 10px;">Cycle</th>
                            <th style="border: 1px solid black; padding: 10px;">Pot Number</th>
                            <th style="border: 1px solid black; padding: 10px;">Actual Yield</th>
                            <th style="border: 1px solid black; padding: 10px;">Target Yield</th>
                            <th style="border: 1px solid black; padding: 10px;">AVG Plant Height</th>
                            <th style="border: 1px solid black; padding: 10px;">AVG Longest Leaf</th>
                            <th style="border: 1px solid black; padding: 10px;">AVG Leaves Count</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            # Add rows dynamically with color logic
            for _, row in df.iterrows():
                html_table += f"""<tr>
                        <td style="border: 1px solid black; padding: 10px;">{row['Cycle']}</td>
                        <td style="border: 1px solid black; padding: 10px; background-color: {pot_number_color(row['Pot Number'])}; color: black;">{row['Pot Number']}</td>
                        <td style="border: 1px solid black; padding: 10px; background-color: {color_cell(row['Actual Yield'], 60, 40)}; color: white;">{row['Actual Yield']}</td>
                        <td style="border: 1px solid black; padding: 10px; background-color: white; color: black;">{row['Target']}</td>
                        <td style="border: 1px solid black; padding: 10px; background-color: {color_cell(row['AVG Plant Height'], 50, 38)}; color: black;">{row['AVG Plant Height']}</td>
                        <td style="border: 1px solid black; padding: 10px; background-color: {color_cell(row['AVG Longest Leaf'], 35, 30)}; color: black;">{row['AVG Longest Leaf']}</td>
                        <td style="border: 1px solid black; padding: 10px; background-color: {color_cell(row['AVG Leaves Count'], 10, 5)}; color: black;">{row['AVG Leaves Count']}</td>
                    </tr>"""

            # Close the table
            html_table += """
                    </tbody>
                </table>
            </div>
            """

            if st.button("More Info"):
                import streamlit as st
                st.write(html_table, unsafe_allow_html=True)


        # Create DataFrame
        df = pd.read_csv('Dataset/Pock choy /generation.csv')
        dfbenchmark = pd.read_csv('Dataset/Benchmark/Pakchoyparameter.csv')

        p = dfbenchmark[dfbenchmark['Is it Important Trait'] == True]['Crop Traits'].iloc[0]
        e = dfbenchmark[dfbenchmark['Is it Important Trait'] == True]['Goal'].iloc[0]
        eh = dfbenchmark[['Goal' , 'Crop Traits']]


        # Filter columns
        filtered_df = df[['generation', 'pot', 'leavescount', 'longestleaf', 'plantheight']]

        # Group by pot and subpot, calculate averages
        grouped_df = filtered_df.groupby(['generation', 'pot']).mean().reset_index()
        # Scale the values based on the height (score)
        max_score = e
        height_scaling = 100
        grouped_df['score'] = grouped_df[p] * height_scaling / max_score

        st.sidebar.markdown(printCostumTitleAndContenth2("Select Season and Plot",
                                                 ""),
                    unsafe_allow_html=True)


        # Read data from CSV file
        df = pd.read_csv('Dataset/Pock choy /PackchoyGeneration2.csv')

        # Convert 'Date' column to datetime format
        # Streamlit app

        selected_pot = st.sidebar.selectbox('Select a Cycle', [1,2,3,4])
        selected_subpot = st.sidebar.selectbox('Select a Plot', [1,2])

        # Filter DataFrame based on selected Pot and SubPot
        filtered_df = df[(df['Pot'] == selected_pot) & (df['SubPot'] == selected_subpot)]
        # Selecting traits
        with st.expander('Traits'):
            import streamlit as st
            import pandas as pd
            import plotly.express as px
            import random

            # Fixed start values for each cycle
            start_values = {
                'Cycle 1': {'AVG Plant Height': 36.0, 'AVG Longest Leaf': 27.0, 'AVG Leaves Count': 6.75},
                'Cycle 2': {'AVG Plant Height': 40.0, 'AVG Longest Leaf': 30.0, 'AVG Leaves Count': 7.5},
                'Cycle 3': {'AVG Plant Height': 44.0, 'AVG Longest Leaf': 33.0, 'AVG Leaves Count': 8.25},
                'Cycle 4': {'AVG Plant Height': 48.0, 'AVG Longest Leaf': 36.0, 'AVG Leaves Count': 9.0}
            }

            # Simulate the dataset for 28 days (gradually changing values for each trait)
            data = []

            for cycle in ['Cycle 1', 'Cycle 2', 'Cycle 3', 'Cycle 4']:
                for pot in [1, 2]:
                    # Generate the trait values over 28 days with gradual changes
                    plant_height = start_values[cycle]['AVG Plant Height']
                    longest_leaf = start_values[cycle]['AVG Longest Leaf']
                    leaf_count = start_values[cycle]['AVG Leaves Count']

                    for day in range(1, 29):  # 28 days
                        # Simulate gradual change over days (increase/decrease)
                        plant_height += random.uniform(0.1, 0.5)  # Gradual increase in plant height
                        longest_leaf += random.uniform(0.05, 0.3)  # Gradual increase in longest leaf
                        leaf_count += random.uniform(0.1, 0.2)  # Gradual increase in leaf count

                        data.append({
                            'Cycle': cycle,
                            'Pot Number': pot,
                            'Day': day,
                            'AVG Plant Height': round(plant_height, 2),
                            'AVG Longest Leaf': round(longest_leaf, 2),
                            'AVG Leaves Count': round(leaf_count, 2)
                        })

            # Create DataFrame
            df = pd.DataFrame(data)

            # Streamlit UI: Title and Cycle Selection
            st.title("Plant Trait Trend Over Time (Simulated Changes)")

            # Dropdown to select the cycle(s)
            selected_cycles = st.multiselect("Select Cycles", df['Cycle'].unique())

            # Dropdown to select the trait(s) to plot
            selected_trait = st.selectbox("Select Plant Trait",
                                          ['AVG Plant Height', 'AVG Longest Leaf', 'AVG Leaves Count'])

            # Filter the data based on selected cycles
            if selected_cycles:
                filtered_df = df[df['Cycle'].isin(selected_cycles)]

                # Plotting the trend of selected plant trait(s) over time
                fig = px.line(filtered_df,
                              x='Day',
                              y=selected_trait,
                              color='Cycle',
                              markers=True,
                              title=f"Trend of {selected_trait} Over Time (Cycles: {', '.join(selected_cycles)})")

                # Show the plot
                st.plotly_chart(fig)

                # Show the data for selected cycles and traits
                st.write("Data for Selected Cycles and Traits:")
                st.dataframe(filtered_df[['Cycle', 'Pot Number', 'Day', selected_trait]])
            else:
                st.write("Please select at least one cycle to view the trend and data.")

        with st.expander('Sensors'):
            import streamlit as st
            import pandas as pd
            import plotly.express as px

            # Fixed dataset for 28 days (you can replace this with your actual data)
            data = {
                'Date': pd.date_range(start="2025-01-01", periods=28, freq="D"),  # 28 days starting from 2025-01-01
                'waterTemperature': [22.5, 23.0, 22.8, 23.3, 24.0, 23.5, 22.9, 23.2, 22.7, 23.1, 22.6, 23.3, 22.9, 23.0,
                                     23.1, 23.4, 22.8, 23.5, 22.7, 23.0, 22.9, 23.1, 23.2, 23.0, 22.8, 23.3, 22.9,
                                     23.2],
                'waterPh': [7.3, 7.4, 7.2, 7.1, 7.3, 7.5, 7.4, 7.6, 7.5, 7.3, 7.2, 7.4, 7.5, 7.6, 7.3, 7.4, 7.5, 7.6,
                            7.2, 7.4, 7.3, 7.5, 7.6, 7.4, 7.3, 7.5, 7.4, 7.6],
                'waterSalinity': [5.5, 5.6, 5.4, 5.3, 5.6, 5.8, 5.7, 5.6, 5.5, 5.6, 5.4, 5.7, 5.8, 5.6, 5.5, 5.7, 5.6,
                                  5.8, 5.7, 5.5, 5.6, 5.7, 5.6, 5.5, 5.7, 5.8, 5.6, 5.7],
                'waterSr': [100, 105, 110, 103, 98, 101, 107, 102, 108, 99, 100, 103, 104, 106, 107, 108, 102, 103, 104,
                            101, 100, 105, 106, 107, 103, 101, 104, 105],
                'waterOrp': [10, 12, 9, 11, 10, 13, 12, 11, 10, 9, 13, 12, 10, 11, 13, 10, 9, 12, 11, 13, 10, 12, 13, 9,
                             11, 10, 12, 13],
                'waterTds': [300, 320, 310, 315, 305, 330, 325, 315, 310, 305, 320, 325, 315, 310, 320, 330, 325, 315,
                             310, 305, 320, 325, 330, 315, 310, 320, 325, 330]
            }

            # Create DataFrame
            df = pd.DataFrame(data)

            # Streamlit UI: Title and Dropdown for selecting multiple water parameters
            st.title("Water Parameter Trend Analysis")

            # Multiselect for selecting multiple water parameters
            parameters = st.multiselect("Select Water Parameters",
                                        ['waterTemperature', 'waterPh', 'waterSalinity', 'waterSr', 'waterOrp',
                                         'waterTds'])

            # Check if at least one parameter is selected
            if parameters:
                # Plotting the trend of the selected water parameters using Plotly
                fig = px.line(df, x='Date', y=parameters, title="Trend of Selected Water Parameters Over Time")
                st.plotly_chart(fig)

                # More Info button to display the data in a table for all selected parameters
                if st.button('Show Tables'):
                    st.write("Data for Selected Parameters:")
                    st.dataframe(df[['Date'] + parameters])
            else:
                st.write("Please select at least one water parameter to view the trend.")



