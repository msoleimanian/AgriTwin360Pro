def exploreConstructor(generation,plot):
    import streamlit as st
    import time
    import numpy as np
    import pandas as pd
    import time
    import plotly.express as px
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import io
    import json
    import requests
    from streamlit_option_menu import option_menu

    def get_field_datas(field_number):
        URL = f'https://api.thingspeak.com/channels/1649792/fields/{field_number}.json?api_key='
        KEY = 'YOUR_API_KEY'
        NEW_URL = URL + KEY

        get_data = requests.get(NEW_URL).json()
        channel_id = get_data['channel']['id']
        field_data = get_data['feeds']
        time = []
        val = []
        df = pd.DataFrame()
        for data in field_data:
            time.append(data['created_at'])
            val.append(data[f'field{field_number}'])

        df['time'] = time
        df['value'] = val
        print(field_data)
        # Extracting values for the specified field
        return df

    def get_field_data(field_number):
        URL = f'https://api.thingspeak.com/channels/1649792/fields/{field_number}.json?api_key='
        KEY = 'YOUR_API_KEY'
        NEW_URL = URL + KEY

        get_data = requests.get(NEW_URL).json()
        channel_id = get_data['channel']['id']
        field_data = get_data['feeds']
        print(field_data[len(field_data) - 1]['created_at'])
        # Extracting values for the specified field
        values = [field_data[1][f'field{field_number}']]
        time = field_data[len(field_data) - 1]['created_at']
        return values

    def getPakChoyData(nutreint):

        url = "https://api.satu.singularityaero.tech/api/telemetries"

        payload = json.dumps({
            "deviceUniqueId": "UPMSO2001",
            "telemetryTypeCode": nutreint,
            "dateStart": "2023-10-10 21:50:39"
        })
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 25|iEjCopmkc73ZFYtCzHCFLnCJb670ErvV3VBfGCt2'
        }

        response = requests.request("GET", url, headers=headers, data=payload)

        data = response.json()
        print("$$$$$$$$$$$$$$$$$$$")
        print(data)
        time = []
        val = []
        dataframe = pd.DataFrame()

        for d in data['data']:
            val.append(d['value'])
            time.append(d['readingAt'])

        dataframe['time'] = time
        dataframe['value'] = val
        print("$$$$$$$$$$$$$$$$$$$")
        return dataframe

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
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: {color}; font-size: 11px; font-weight: bold;">{label}<br>{value} </div>
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

    # Display the animated line using HTML
    option2 = 'Pak choy'
    if option2 == 'Pak choy':

        col1, col2 = st.columns(2)

        with col1:
            option = generation

        with col2:
            optionplot = plot

        dfbest = pd.read_csv(f'Dataset/Pock choy /PackchoyGeneration{option}.csv')
        # Calculate the day count for each unique date
        dfbest['Day'] = dfbest['Date'].rank(method='dense').astype(int)
        # Keep only the relevant columns
        dfbest = dfbest[['Day', 'Pot', 'SubPot', 'EC', 'pH', 'Leaves Count', 'Longest Leaf', 'Plant Height(mm)']]
        # Display the modified DataFrame
        print("####################")
        print(dfbest)
        nutrients = ['EC', 'pH']
        fnbest2 = dfbest.query(f"""Day == 2""")
        fnbest3 = dfbest.query(f"""Day == 3""")
        fnbest4 = dfbest.query(f"""Day == 4""")
        fnbest5 = dfbest.query(f"""Day == 5""")
        fnbest6 = dfbest.query(f"""Day == 6""")
        fnbest7 = dfbest.query(f"""Day == 7""")
        fnbest8 = dfbest.query(f"""Day == 8""")
        fnbest9 = dfbest.query(f"""Day == 9""")
        fnbest10 = dfbest.query(f"""Day == 10""")
        fnbest11 = dfbest.query(f"""Day == 11""")
        for i in range(2):
            n = nutrients[i]
            nutrient_data = {f'{n} generation{option} pot{optionplot}': [fnbest2[nutrients[i]].values[0],
                                                                         fnbest3[nutrients[i]].values[0],
                                                                         fnbest4[nutrients[i]].values[0],
                                                                         fnbest5[nutrients[i]].values[0],
                                                                         fnbest6[nutrients[i]].values[0],
                                                                         fnbest7[nutrients[i]].values[0],
                                                                         fnbest8[nutrients[i]].values[0],
                                                                         fnbest9[nutrients[i]].values[0],
                                                                         fnbest10[nutrients[i]].values[0],
                                                                         fnbest11[nutrients[i]].values[0]]}

            # Create a DataFrame with the dictionary
            df = pd.DataFrame.from_dict(nutrient_data, orient='index', columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

            # Animated line chart with Plot
            fig = px.line(df.transpose(), x=df.columns, y=df.index,
                          labels={'value': 'Nutrient Level', 'variable': 'Nutrient'},
                          title=n)
            fig.update_layout(xaxis_title='DAYS')
            fig.update_traces(mode='lines+markers')

            # Display the animated chart
            st.plotly_chart(fig)




        html_content = f"""
                                    <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                                        <h3 style="color:#333333;">Query</h3>
                                                        <p>You can see the trend of the metric base on the date that you can select.</p>
                                                    </div>
                                                """
        #   st.markdown(html_content, unsafe_allow_html=True)


        col_1, col_2, col_3 = st.columns(3)
        df = pd.DataFrame()
        with col_1:
            optionsnutrient = st.selectbox(
                'Select the metric',
                ('waterTemperature', 'waterPh', 'waterSr', 'waterOrp', 'waterTds', 'waterSalinity'))
            df = getPakChoyData(optionsnutrient)
        with col_2:
            startDate = st.selectbox(
                'Select the Start Date',
                (df['time'])
            )
        with col_3:
            endDate = st.selectbox(
                'Select the End Date',
                (df['time'])
            )
        mask = (df['time'] > startDate) & (df['time'] <= endDate)
        p = px.line(df.loc[mask], x='time', y='value', title=optionsnutrient)
        st.plotly_chart(p)


        st.write('Data')
        l = ['waterTemperature', 'waterPh', 'waterSr', 'waterOrp', 'waterTds', 'waterSalinity', 'waterEc']
        for p in l:
            df = getPakChoyData(p)
            st.write(p)
            mask = (df['time'] > '2023-10-10 22:46:13') & (df['time'] <= '2023-11-11 23:58:26')
            st.write(df.loc[mask])
            print('################################################')
            df1 = df
            df1['time'] = pd.to_datetime(df['time'])
            df1['value'] = pd.to_numeric(df['value'])

            # Group by date and calculate the mean
            result_df = df1.groupby(df1['time'].dt.date)['value'].mean().reset_index()

            st.write(result_df)
            print('################################################')


    if option2 == 'Aqua':
        selectdate = st.selectbox(
            "Select the date...",
            ("22-Aug-23", "29 - Aug - 23", "5 - Sep - 23", "12 - Sep - 23", "19 - Sep - 23", "26 - Sep - 23"),
            index=0,
            placeholder="Select the date...",
        )
        st.markdown(printCostumTitleAndContenth2(f'Date {selectdate}', ''), unsafe_allow_html=True)
        st.markdown(printCostumTitleAndContenth3('Growth Traits', ''), unsafe_allow_html=True)

        df = pd.read_csv(f'Dataset/Aqua/AquaSammury.csv')
        col1, col2 = st.columns(2)
        with col1:
            df = df.query(f"Date == '{selectdate}'")
            st.write("High Value Trait")
            max_weight = 2  # Maximum weight in KG
            print(df['WEIGHT (kg)'])
            current_weight = df['WEIGHT (kg)'].mean()  # Current weight in KG
            progress_html = animated_circular_progress_bar('WEIGHT (kg)', current_weight, max_weight,
                                                           color='green',
                                                           max_size=200)
            st.components.v1.html(progress_html, height=210)

        with col2:
            df = df.query(f"Date == '{selectdate}'")
            st.write("High Value Trait")
            max_weight = 57.80  # Maximum weight in KG
            current_weight = df['LENGTH (cm)'].mean()  # Current weight in KG
            progress_html = animated_circular_progress_bar('LENGTH (cm)', current_weight, max_weight,
                                                           color='green',
                                                           max_size=200)
            st.components.v1.html(progress_html, height=210)

        fieldname = {'Field 1': 'Temperature', 'Field 2': 'pH', 'Field 3': 'Ammonia', 'Field 4': 'DO',
                     'Field 5': 'Salinity'}
        all_data = {fieldname[f'Field {i}']: get_field_data(i) for i in range(1, 6)}
        print(all_data)
        print(float(all_data['pH'][0]))

        # ---------------------------------------------------------------------------------

        URL = f'https://api.thingspeak.com/channels/1649792/fields/1.json?api_key='
        KEY = 'YOUR_API_KEY'
        NEW_URL = URL + KEY

        get_data = requests.get(NEW_URL).json()
        channel_id = get_data['channel']['id']
        field_data = get_data['feeds']
        print(field_data[len(field_data) - 1]['created_at'])

        html_content = f"""
                                    <div style="border: 2px solid #333333; padding:10px; border-radius:5px;">
                                                        <h3 style="color:#333333;">Query</h3>
                                                        <p>You can see the trend of the metric base on the date that you can select.</p>
                                                    </div>
                                                """
        st.markdown(html_content, unsafe_allow_html=True)

        col_1, col_2, col_3 = st.columns(3)
        df = pd.DataFrame()
        with col_1:
            options = st.selectbox(
                'Select the option',
                ('Temperature', 'pH', 'Ammonia', 'DO', 'Salinity'))

            fieldname2 = {'Temperature': 1, 'pH': 2, 'Ammonia': 3, 'DO': 4, 'Salinity': 5}
            print(get_field_datas(fieldname2[options]))
            df = get_field_datas(fieldname2[options])
        with col_2:
            startDate = st.selectbox(
                'Select the Start Date',
                (df['time'])
            )
        with col_3:
            endDate = st.selectbox(
                'Select the End Date',
                (df['time'])
            )
        mask = (df['time'] > startDate) & (df['time'] <= endDate)
        p = px.line(df.loc[mask], x='time', y='value', title=options)
        st.plotly_chart(p)

