import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import pandas as pd
import time

import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def printCostumTitleAndContenth3(title, context):
    return f"""
        <div class="jumbotron">
        <h3>{title}</h3>
        <h6>{context}</h6>
        </div>
        <div class="container">
        </div>
        """


def printWithTitleAndBoarder(title, context):
    return f"""
        <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
            <h4 style="color:#333333;">{title}</h4>
            <table>
                    <tr>
                        <th>Status</th>
                        <th>Score(10)</th>
                        <th>Risk Level</th>
                    </tr>
                    <tr>
                        <td style="color : red;">good</td>
                        <td style="color : red;">4</td>
                        <td style="color : red;">High Risk</td>
                    </tr>
            </table>
            <h4 style="color:#333333;">Health Status of the Pots</h4>
            <table>
                    <tr>
                        <th>Pot Number</th>
                        <th>Predicted Crop Health Status</th>
                        <th>Yeild Report</th>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td style="color : red;">bad</td>
                        <td style="color : red;">Predicted Average Weight for Generation 3 at Week4: 670 gram (% 39.09 lower than the best, Best weight grain is 1100 gram.)</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td style="color : red;">bad</td>
                        <td style="color : red;">Predicted Average Weight for GenerationGeneration 3 at Week4: 690 gram (% 37.27 lower than the best, Best weight grain is 1100 gram.)</td>
                    </tr>
            </table>
        </div>
        """


def cardCreator(title, value):
    html_code = """
    <html>
        <style>
            .status-card {
                background-color: #c1f0c1; /* Light green color */
                padding: 20px;
                border-radius: 50%; /* Make it circular */
                width: 170px; /* Set a fixed width for the circular card */
                height: 170px; /* Set a fixed height for the circular card */
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
            }
        </style>"""

    html_code = html_code + f"""<div class='status-card'>
            <h5>{title}</h5>
            <p> {value} </p>
        </div>
        </html>
    """
    st.markdown(html_code, unsafe_allow_html=True)


def animated_gauge_progress_bar(value, title, rmin, rmax):
    if 1 <= value <= 10:
        bar_color = "red"
    else:
        bar_color = "#4CAF50"  # Default color for other values

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{'type': 'indicator'}]]
    )
    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 50, 'position': 'top'},
        gauge=dict(
            axis=dict(range=[rmin, rmax]),
            bar=dict(color='black'),  # Set color dynamically
            bgcolor="white",
            borderwidth=2,
            bordercolor="gray",
            steps=[dict(range=[0, 100], color="lightgray")]
        ),
        title=dict(text=title, font=dict(size=20)),  # Set title directly within the trace
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=15, r=15, b=15, t=60),
    )

    return fig


def guageCreator(vlaue, title, rmin, rmax):
    # Streamlit app
    chart_placeholder = st.empty()
    vla = round(vlaue)
    if vla <= 1:
        animated_chart = animated_gauge_progress_bar(vlaue, title, rmin, rmax)
        chart_placeholder.plotly_chart(animated_chart, use_container_width=True)
    # Update the progress value with an animation
    else:
        for value in range(0, vla, 1):
            animated_chart = animated_gauge_progress_bar(value, title, rmin, rmax)
            chart_placeholder.plotly_chart(animated_chart, use_container_width=True)
            st.empty()  # Clear the previous chart to create animation effect


def textwithboarder(title, text):
    html_content = f"""
            <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                <h3 style="color:#333333;">{title}</h3>
                                <p>{text}</p>
                            </div>"""
    # Show the Guage of the Nutrients levels
    st.markdown(html_content, unsafe_allow_html=True)


def printWithTitleAndBoarderwithoutTable(title, context):
    return f"""
        <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
            <h4 style="color:#333333;">{title}</h4>
            <table>
                    <tr>
                        <th>Status</th>
                        <th>Score(10)</th>
                    </tr>
                    <tr>
                        <td style="color : red;">bad</td>
                        <td style="color : red;">5</td>
                    </tr>
            </table>
            <h4 style="color:#333333;">Plant Cycle</h4>
            <table>
                    <tr>
                        <th>Current Growth Time</th>
                        <th>Total Germination Time</th>
                    </tr>
                    <tr>
                        <td>2 week</td>
                        <td>4 week</td>
                    </tr>
            </table>
            <h4 style="color:#333333;">Health Status of the Pots</h4>
            <table>
                    <tr>
                        <th>Pot Number</th>
                        <th>Current Crop Health Status</th>
                    </tr>
                    <tr>
                        <td>1</td>
                        <td style="color : red;">bad</td>
                    </tr>
                    <tr>
                        <td>2</td>
                        <td style="color : orange;">normal</td>
                    </tr>
            </table>
        </div>
        """


def printWithTitleAndBoarder1(title, context):
    return f"""
        <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
            <h3 style="color:#333333;">{title}</h3>
            <h5>{context}</h5>
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

def SimulationConstructor():
    import pandas as pd
    import numpy as np


    option2 = option_menu(None, ["Pak choy", "Rice", "Aqua"],
                          menu_icon="forward", default_index=0, orientation="horizontal",
                          styles={
                              "container": {"padding": "0!important", "background-color": "#fafafa"},
                              "icon": {"color": "orange", "font-size": "15px"},
                              "nav-link": {"font-size": "15px", "text-align": "right", "margin": "0px",
                                           "--hover-color": "#eee", },
                              "nav-link-selected": {"background-color": "green"},
                          }
                          )
    if option2 == "Pak choy":

        option3 = option_menu(None, ["Crop Information Simulation", "Yield Simulation"],
                              menu_icon="forward", default_index=0, orientation="horizontal",
                              styles={
                                  "container": {"padding": "0!important", "background-color": "#fafafa"},
                                  "icon": {"color": "orange", "font-size": "15px"},
                                  "nav-link": {"font-size": "15px", "text-align": "right", "margin": "0px",
                                               "--hover-color": "#eee", },
                                  "nav-link-selected": {"background-color": "blue"},
                              }
                              )


        if option3 == "Crop Information Simulation":
            #***********************
            st.markdown(printWithTitleAndBoarder1('Simulation' , """""") , unsafe_allow_html=True)
            st.write('')
            col1, col2 = st.columns(2)
            temperature = col1.slider("Temperature (°C)", min_value=0.0, max_value=40.0, step=0.1, value=25.0)
            tds = col1.slider("TDS (ppm)", min_value=0.0, max_value=25.0, step=0.1, value=10.0)
            orp = col2.slider("ORP (mV)", min_value=40.0, max_value=100.0, step=0.1, value=70.0)
            sr = col2.slider("Sr", min_value=0.0, max_value=2.0, step=0.1, value=1.0)
            ec = col2.slider("EC (µS/cm)", min_value=0.0, max_value=20.0, step=0.1, value=10.0)
            ph = col1.slider("pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0)
            day = col1.slider("Day", min_value=10.0, max_value=31.0, step=1.0, value=15.0)

            potnumber = col2.selectbox('Select the Pot:', (1, 2))
            import requests
            import json

            url = 'https://cb70-34-16-142-124.ngrok-free.app/LeavesCount_prediction'

            # Replace with your Ngrok URL
            potnumberlist = []
            leafcountnumber = []
            for i in range(40):
                potnumberlist.append(i+1)


            df = pd.DataFrame()
            input_data_for_model = {
                'Pot' : potnumber,
                'waterTemperature': temperature,
                'waterPh': ph,
                'waterSr': sr,
                'waterOrp': orp,
                'waterTds': tds,
                'waterEc': ec,
                'Day': day
            }

            response = requests.post(url, json=input_data_for_model)
            if response.status_code == 200:
                prediction = response.json()
                print((prediction['prediction']))
                df['leafcount'] = (prediction['prediction'])
                df['subpotnumber'] = potnumberlist
            else:
                print("Error:", response.text)
            #***********************************

            # Define a mapping for status to color
            status_color_mapping = {'good': 'green', 'normal': 'orange', 'bad': 'red'}

            # Create a dictionary with pot number as keys and color as values
            pot_color_dict = {}

            def update_status(leaf_count, benchmark=10):
                if day/2 - 1 <= leaf_count <= day/2:
                    return 'Good'
                elif day/2 - 5 <= leaf_count <= day/2 - 3:
                    return 'Good'
                else:
                    return 'Good'

            df['status'] = df['leafcount'].apply(update_status)
            for index, row in df.iterrows():
                pot_number = row['subpotnumber']
                status = row['status'].lower()  # Convert to lowercase for case-insensitivity
                color = status_color_mapping.get(status,
                                                 'unknown')  # Default to 'unknown' if status is not one of the specified values
                pot_color_dict[pot_number] = color
            dataframe = pot_color_dict

            css_styles = """
                                <style>
                                    .button-container {
                                        display: grid;
                                        grid-template-columns: repeat(8, 1fr);
                                        gap: 10px;
                                        border: 2px solid #ddd; /* Border around the button container */
                                        padding: 10px; /* Add some padding for better appearance */
                                    }
                                    .button {
                                        width: 100%;
                                        height: 70px;
                                        background-color: #eee;
                                        color: #333;
                                        font-size: 20px;
                                        font-weight: bold;
                                        border: 2px solid #ddd;
                                        border-radius: 55px;
                                        cursor: pointer;
                                    }
                                </style>
                            """
            # Display CSS styles
            st.markdown(css_styles, unsafe_allow_html=True)

            # Create button grid
            button_container = f"""<div style="border: 2px solid #333333; padding:10px; border-radius:5px;">     <p style='text-align: center'>Pot  1</p>  <div class='button-container'>"""
            for key, value in dataframe.items():
                # Use Streamlit's button widget with a callback to display text on click
                button_container += f"""<button class='button' style='background-color: {value}; border-color: {value}; color: white'
                                                   onclick='st.write("{key} clicked!")'>{key}</button>"""
            button_container += "</div> </div>"

            col1 , col2 = st.columns(2)

            # Display button grid
            with col1:
                st.markdown(button_container, unsafe_allow_html=True)

            with col2:
                #***********************************
                df2 = pd.read_csv('Dataset/Pock choy /Generation3_pot1_Simulation.csv')
                dataframe = df2
                selectpot = st.selectbox('Select the SubPot Number', range(1, 41))
                subpot = f"SubPot{selectpot}"
                filtered_df = df2[df2['subpotnumber'] == subpot]

                html_content = f"""
                                        <div style="border: 5px solid #333333; padding:10px; border-radius:5px;">
                                            <h4 style="color: {filtered_df['subpotnumber']}; ">Crop Status: good</h2>
                                            <h4>Current Crop Traits</h2>
                                            <div style="overflow-x:auto;">
                                            <table>
                                                <tr>
                                                    <th>Leaf Count</th>
                                                    <th>Longest Leaf (mm)</th>
                                                    <th>Plant Height (mm)</th>
                                                </tr>
                                                <tr>
                                                    <td>{filtered_df['leafcount'].values[0]}</td>
                                                    <td>{filtered_df['longestleaf'].values[0]}</td>
                                                    <td>{filtered_df['plantheight'].values[0]}</td>
                                                </tr>
                                            </table>
                                            
                                        """
                html_content = f"""
                                                <div style="border: 5px solid #333333; padding:10px; border-radius:5px;">
                                                    <h4 style="color: {filtered_df['subpotnumber']}; ">Crop Status in future: Good</h2>
                                                    <table border='1'>
                                                        <tr>
                                                            <th>Leaf Count</th>
                                                            <th>Longest Leaf (mm)</th>
                                                            <th>Plant Height (mm)</th>
                                                        </tr>
                                                        <tr>
                                                            <td>{round(df['leafcount'].values[0],0) }</td>
                                                            <td>{filtered_df['longestleaf'].values[0] + 22}</td>
                                                            <td>{filtered_df['plantheight'].values[0] + 44}</td>
                                                            <td> Predicted Traits at day{day} </td>
                                                        </tr>
                                                        <tr>
                                                            <td>{round(df['leafcount'].values[0],0)+5 }</td>
                                                            <td>{filtered_df['longestleaf'].values[0] + 66}</td>
                                                            <td>{filtered_df['plantheight'].values[0] + 75}</td>
                                                            <td> Target Traits at day{day} </td>
                                                        </tr>
                                                    </table>
                                                </div>
                                                """
                col2.markdown(html_content, unsafe_allow_html=True)

        if option3 == "Yield Simulation":
            import matplotlib.pyplot as plt

            st.title("Yield Simulation")

            # Slider to select the target yield per container
            target_yield = st.slider("Select Target Yield (grams/container)", min_value=650, max_value=1600, step=50,
                                     value=1000)

            st.write(f"Selected Target Yield: {target_yield} grams/container")

            # Simulate the environmental parameters from day 10 to day 27
            days = np.arange(10, 28)
            np.random.seed(42)  # For consistent random data

            def simulate_parameters(target_yield, days):
                trend_factor = (target_yield - 1000) / 1000.0  # Adjust trend factor to the new default of 1000

                # Simulate pH: Decreasing trend with reduced fluctuations
                ph = 7.5 - 0.03 * trend_factor * (days - 10) + np.random.normal(0, 0.05, len(days))

                # Simulate EC: Increasing trend with reduced fluctuations
                ec = 0.5 + 0.05 * trend_factor * (days - 10) + np.random.normal(0, 0.05, len(days))

                # Simulate Temperature: Increasing trend with reduced fluctuations
                temperature = 20 + 0.3 * trend_factor * (days - 10) + np.random.normal(0, 0.2, len(days))

                # Simulate TDS: Increasing trend with reduced fluctuations
                tds = 0.5 + 0.03 * trend_factor * (days - 10) + np.random.normal(0, 0.1, len(days))

                return ph, ec, temperature, tds

            ph, ec, temperature, tds = simulate_parameters(target_yield, days)

            # Plotting the parameters
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            # pH
            axs[0, 0].plot(days, ph, marker='o', color='blue')
            axs[0, 0].set_title("pH Level Over Time")
            axs[0, 0].set_xlabel("Days")
            axs[0, 0].set_ylabel("pH")

            # EC
            axs[0, 1].plot(days, ec, marker='o', color='green')
            axs[0, 1].set_title("EC Level Over Time")
            axs[0, 1].set_xlabel("Days")
            axs[0, 1].set_ylabel("EC (µS/cm)")

            # Temperature
            axs[1, 0].plot(days, temperature, marker='o', color='red')
            axs[1, 0].set_title("Temperature Over Time")
            axs[1, 0].set_xlabel("Days")
            axs[1, 0].set_ylabel("Temperature (°C)")

            # TDS
            axs[1, 1].plot(days, tds, marker='o', color='purple')
            axs[1, 1].set_title("TDS Level Over Time")
            axs[1, 1].set_xlabel("Days")
            axs[1, 1].set_ylabel("TDS (ppm)")

            plt.tight_layout()
            st.pyplot(fig)



SimulationConstructor()