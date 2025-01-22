

def startai():
    import streamlit as st

    selected_farm = st.sidebar.selectbox("Select the farm:", ["Crop Health Detection and Reporting", "Machine Learning","Yield Simulation"])

    if selected_farm == 'Crop Health Detection and Reporting':

        import cv2
        import numpy as np
        from PIL import Image
        import streamlit as st

        import streamlit as st

        # HTML content for instructions
        html_content = """
        <div style="background-color:#f0f8ff; padding:15px; border-radius:10px; border:1px solid #d3d3d3;">
            <h2 style="color:#007acc;">Welcome to the Crop Analysis Dashboard</h2>
            <p>This tool allows you to upload an image, select the crop type, choose the analysis method, define the actions you'd like to perform, and set the crop growth timeline. Follow the steps below:</p>
            <ol>
                <li><h4 style="color:#007acc;">Step 1: Upload an Image</h4></li>
                <li><h4 style="color:#007acc;">Step 2: Select the Crop Type</h4></li>
                <li><h4 style="color:#007acc;">Step 3: Choose the Analysis Method</h4></li>
                <li><h4 style="color:#007acc;">Step 4: Select the Analysis Options</h4></li>
                <li><h4 style="color:#007acc;">Step 5: Set the Growth Timeline</h4></li>
            </ol>
        </div>
        """

        # Display HTML instructions
        st.markdown(html_content, unsafe_allow_html=True)

        # Step 1: File upload
        st.write("### Step 1: Upload an Image")
        uploaded_file = st.file_uploader("Upload an image of the crop:", type=["jpg", "jpeg", "png"])

        # Step 2: Select the crop type
        st.write("### Step 2: Select the Crop Type")
        crop_type = st.selectbox("Select the type of crop:", ["Pak Choy", "Rice"])

        # Step 3: Choose the analysis method
        st.write("### Step 3: Choose the Analysis Method")
        analysis_method = st.radio("Select the analysis method:", ["CNN", "OpenCV"])

        # Step 4: Select the analysis options
        st.write("### Step 4: Select the Analysis Options")
        analyze_health = st.checkbox("Analyze Health")
        estimate_yield = st.checkbox("Estimated Yield")
        pest_detection = st.checkbox("Pest Detection")

        # Step 5: Set the growth timeline
        st.write("### Step 5: Set the Growth Timeline")
        total_weeks = st.text_input("Enter the total number of weeks needed for growth:", placeholder="e.g., 12")
        current_week = st.text_input("Enter the current week number:", placeholder="e.g., 6")

        # Display the summary of user inputs
        st.write("### Summary of Your Selections")

        if uploaded_file:
            st.write(f"**Uploaded File**: {uploaded_file.name}")
        else:
            st.write("No image uploaded yet.")

        st.write(f"**Selected Crop Type**: {crop_type}")
        st.write(f"**Selected Analysis Method**: {analysis_method}")

        st.write("**Selected Analysis Options:**")
        if analyze_health:
            st.write("- Analyze Health")
        if estimate_yield:
            st.write("- Estimated Yield")
        if pest_detection:
            st.write("- Pest Detection")
        if not (analyze_health or estimate_yield or pest_detection):
            st.write("No options selected.")

        # Display the growth timeline summary
        if total_weeks and current_week:
            try:
                total_weeks = int(total_weeks)
                current_week = int(current_week)
                if current_week > total_weeks:
                    st.error("The current week cannot exceed the total weeks needed for growth.")
                else:
                    remaining_weeks = total_weeks - current_week
                    st.write(
                        f"**Growth Timeline:** The crop needs {total_weeks} weeks to grow. You are currently in week {current_week}.")
                    st.write(f"**Remaining Weeks:** {remaining_weeks} weeks left for growth.")
            except ValueError:
                st.error("Please enter valid numbers for the total weeks and current week.")
        else:
            st.write("**Growth Timeline:** Not fully provided yet.")

        def detect_crops_and_health_week2(image, min_pixels=900, min_health_percentage=40, min_component_area=500):
            """
            Detect crops and analyze their health based on green pixels in the image.

            Parameters:
                image (numpy.ndarray): The input image (RGB or BGR format).
                min_pixels (int): Minimum green pixels to classify a crop as healthy.
                min_health_percentage (float): Minimum percentage of green pixels for a crop to be considered healthy.
                min_component_area (int): Minimum area (in pixels) for a detected crop.

            Returns:
                processed_image (numpy.ndarray): The input image with crops annotated.
                crop_health (list): A list of dictionaries containing health data for each crop.
                healthy_crops (int): The number of healthy crops detected.
                total_crops (int): The total number of crops detected.
                improved_mask (numpy.ndarray): Binary mask highlighting green areas in the image.
            """

            # Ensure the input image is a numpy array
            if not isinstance(image, np.ndarray):
                raise ValueError("Input image must be a numpy array.")

            # Convert to BGR format if the image is in RGB
            if len(image.shape) == 3 and image.shape[2] == 3:  # Check for color image
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Convert to HSV for color-based segmentation
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define green color range for segmentation
            lower_green = np.array([35, 40, 40])
            upper_green = np.array([90, 255, 255])

            # Create a mask for green areas
            green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

            # Morphological operations to remove noise and close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            cleaned_green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

            # Detect connected components in the cleaned mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_green_mask, connectivity=8)

            # Initialize variables for crop health analysis
            processed_image = image.copy()
            crop_health = []
            total_crops = 0
            healthy_crops = 0

            # Analyze each detected component
            for i in range(1, num_labels):  # Skip the background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_component_area:  # Filter components by size
                    total_crops += 1

                    # Get bounding box coordinates
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
                        i, cv2.CC_STAT_HEIGHT]

                    # Extract the crop mask and calculate green pixel percentage
                    crop_mask = cleaned_green_mask[y:y + h, x:x + w]
                    green_pixels = cv2.countNonZero(crop_mask)
                    total_pixels = w * h
                    green_percentage = (green_pixels / total_pixels) * 100

                    # Determine health status
                    if green_percentage >= min_health_percentage and green_pixels >= min_pixels:
                        health_status = "Healthy"
                        healthy_crops += 1
                        border_color = (0, 255, 0)  # Green for healthy crops
                    else:
                        health_status = "Unhealthy"
                        border_color = (255, 0, 0)  # Red for unhealthy crops

                    # Annotate the processed image
                    cv2.rectangle(processed_image, (x, y), (x + w, y + h), border_color, 3)
                    cv2.putText(processed_image, f"Crop {total_crops}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                border_color, 2)

                    # Append crop health data
                    crop_health.append({
                        "Crop": total_crops,
                        "Health Status": health_status,
                        "Green Pixels": green_pixels,
                        "Total Pixels": total_pixels,
                        "Green Percentage": green_percentage
                    })

            # Create a mask highlighting green areas
            improved_mask = np.zeros_like(image)
            improved_mask[cleaned_green_mask > 0] = [0, 255, 0]

            return processed_image, crop_health, healthy_crops, total_crops, improved_mask


        def generate_html_report(crop_health, healthy_crops, total_crops, desired_yield=250):
            """
            Generate an HTML report for crop health analysis.

            Parameters:
                crop_health (list): List of dictionaries containing crop health data.
                healthy_crops (int): Number of healthy crops detected.
                total_crops (int): Total number of crops detected.
                desired_yield (int): Desired yield for the area (in grams).

            Returns:
                str: HTML content for the report.
            """
            healthy_percentage = (healthy_crops / total_crops) * 100 if total_crops > 0 else 0

            # Create a header for the overall summary
            html_content = f"""<div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; background-color: #f9f9f9;">
                <h4 style="color: #4CAF50; text-align: center;">Health Status of Area</h4>
                <p><b>Total Crops Detected:</b> {total_crops}</p>
                <p><b>Healthy Crops:</b> {healthy_crops}/{total_crops} ({healthy_percentage:.2f}%)</p>
                <p><b>Target Yield for the Area:</b> {desired_yield} grams</p>
            </div>"""

            # Add a table for detailed crop data
            html_content += """<table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                <tr style="background-color: #4CAF50; color: white;">
                    <th style="border: 1px solid #ddd; padding: 8px;">Crop</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Health Status</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Green Pixels</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Total Pixels</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Green Percentage (%)</th>
                    <th style="border: 1px solid #ddd; padding: 8px;">Canopy Size (mm²)</th>
                </tr>"""

            # Add rows for each crop
            for crop in crop_health:
                canopy_size = crop["Total Pixels"] * 0.1  # Assuming each pixel represents 0.1 mm²
                html_content += f"""<tr>
                    <td style="border: 1px solid #ddd; padding: 8px;">Crop {crop['Crop']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{crop['Health Status']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{crop['Green Pixels']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{crop['Total Pixels']}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{crop['Green Percentage']:.2f}</td>
                    <td style="border: 1px solid #ddd; padding: 8px;">{canopy_size:.2f} mm²</td>
                </tr>"""

            # Close the table
            html_content += "</table>"

            return html_content


        import streamlit as st
        from PIL import Image
        import numpy as np

        # File uploader to upload an image
        uploaded_image = uploaded_file

        if uploaded_image is not None:
            # Open the uploaded image using PIL
            image = Image.open(uploaded_image)

            # Convert the PIL image to a numpy array
            image_array = np.array(image)

            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Call your crop health detection function here
            processed_image, crop_health, healthy_crops, total_crops, improved_mask = detect_crops_and_health_week2(
                image_array)

            # Generate the HTML report
            html_report = generate_html_report(crop_health, healthy_crops, total_crops)

            # Display the report
            st.markdown(html_report, unsafe_allow_html=True)

            # Optionally display the processed image and mask
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            st.image(improved_mask, caption="Improved Mask", use_column_width=True)
        else:
            st.write("Please upload an image to analyze.")

    if selected_farm == 'Machine Learning':
        import streamlit as st
        import pandas as pd
        import time
        import random

        st.markdown(
            """
            <div style="background-color:#f0f8ff; padding:15px; border-radius:10px; border:1px solid #d3d3d3;">
                <h2 style="color:#007acc;">Welcome to the Simulated ML Dashboard.</h2>
                <p>This tool allows you to simulate the process of training and tuning Machine Learning models. Here's how it works:</p>
                <ol>
                    <li><strong>Step 1:</strong> Upload your dataset in CSV format.</li>
                    <li><strong>Step 2:</strong> Select the input features and target column.</li>
                    <li><strong>Step 3:</strong> Choose models and a tuning method (Grid Search or Random Search).</li>
                    <li><strong>Step 4:</strong> Run the simulation to view the best parameters and performance metrics.</li>
                </ol>
                <p style="color: #333;">This dashboard is for demonstration purposes and does not perform actual training.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Mock model performance and parameter results
        mock_tuning_results = {
            'RandomForest': {'best_params': {'n_estimators': 100, 'max_depth': 10}, 'accuracy': 0.91},
            'AdaBoost': {'best_params': {'n_estimators': 200, 'learning_rate': 0.1}, 'accuracy': 0.89},
            'KNN': {'best_params': {'n_neighbors': 5}, 'accuracy': 0.85},
            'MLP': {'best_params': {'hidden_layer_sizes': (100,), 'max_iter': 500}, 'accuracy': 0.88},
            'LogisticRegression': {'best_params': {'C': 1.0, 'penalty': 'l2'}, 'accuracy': 0.86},
            'DecisionTree': {'best_params': {'max_depth': 10, 'min_samples_split': 2}, 'accuracy': 0.84},
            'SVM': {'best_params': {'C': 1.0, 'kernel': 'rbf'}, 'accuracy': 0.87},
            'AutoML_H2O': {'best_params': 'Auto-tuned by H2O', 'accuracy': 0.92},
            'AutoML_TPOT': {'best_params': 'Auto-tuned by TPOT', 'accuracy': 0.90},
            'AutoML_AutoSklearn': {'best_params': 'Auto-tuned by Auto-Sklearn', 'accuracy': 0.91}
        }


        st.info("Please follow the steps below to proceed.")

        # Step 1: Dataset Upload
        st.write("### Step 1: Upload Dataset")
        uploaded_file = st.file_uploader("Upload your CSV file to get started:", type=["csv"])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            # Display Dataset Preview
            st.write("### Dataset Preview")
            st.dataframe(df.head(10))  # Display the first 10 rows of the dataset

            # Step 2: Feature and Target Selection
            st.write("### Step 2: Select Features and Target")
            columns = df.columns.tolist()

            # Select Features
            features = st.multiselect("Select Feature Columns", options=columns,
                                      help="These columns will be used as input features.")
            # Update Target Options
            target_options = [col for col in columns if col not in features]
            target = st.selectbox("Select Target Column", options=target_options,
                                  help="This column will be the output/target variable.")

            if features and target:
                X = df[features]
                y = df[target]

                # Step 3: Model Selection and Tuning Method
                st.write("### Step 3: Select ML Models and Tuning Method")
                model_options = [
                    'RandomForest', 'AdaBoost', 'KNN', 'MLP',
                    'LogisticRegression', 'DecisionTree', 'SVM',
                    'AutoML_H2O', 'AutoML_TPOT', 'AutoML_AutoSklearn'
                ]
                selected_models = st.multiselect("Select Models to Simulate", options=model_options,
                                                 help="Select one or more models for simulation.")
                tuning_method = st.radio("Select Tuning Method", options=['Grid Search', 'Random Search'],
                                         help="Choose the hyperparameter tuning method.")

                if selected_models:
                    # Step 4: Simulate Model Training and Tuning
                    st.write("### Step 4: Simulate Model Training and Tuning")
                    simulate_button = st.button("Simulate Training and Tuning")

                    if simulate_button:
                        st.write("Preparing Data...")
                        progress = st.progress(0)
                        time.sleep(1)  # Simulate data preparation
                        progress.progress(20)

                        st.write(f"Simulating Model Training and Tuning using {tuning_method}...")
                        tuned_results = {}
                        for i, model_name in enumerate(selected_models):
                            time.sleep(2)  # Simulate tuning time for each model
                            progress.progress(20 + (i + 1) * int(80 / len(selected_models)))

                            # Simulate tuning results
                            tuned_results[model_name] = mock_tuning_results[model_name]

                        progress.progress(100)
                        st.success("Simulation Complete!")

                        # Display Tuned Results
                        st.write("### Tuned Model Results")
                        for model_name, result in tuned_results.items():
                            st.write(f"**{model_name}**")
                            st.write(f"- Best Parameters: {result['best_params']}")
                            st.write(f"- Accuracy: {result['accuracy']:.2f}")

        else:
            st.warning("Please upload a dataset to proceed.")

    if selected_farm == 'Yield Simulation':
        import numpy as np
        import matplotlib.pyplot as plt
        import streamlit as st

        html_code = """
        <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; border:1px solid #d3d3d3;">
            <h2 style="color:#007acc;">Welcome to the Simulated ML Dashboard</h2>
            <p style="color:#333; font-size:16px; line-height:1.6;">
                The <strong>Yield Simulation</strong> feature allows users to input their desired yield goals, and the system
                will dynamically display the corresponding nutrient values required over time. This simulation helps in
                visualizing the relationship between yield goals and nutrient management, enabling data-driven decisions.
            </p>
            <p style="color:#333; font-size:16px; line-height:1.6;">
                Here’s how it works:
            </p>
            <ul style="color:#333; font-size:16px; line-height:1.6;">
                <li>Input your desired yield target in the simulation panel.</li>
                <li>The system will calculate and display the nutrient values over a simulated timeline.</li>
                <li>Observe the trends and adjust your yield goals to optimize nutrient usage.</li>
            </ul>
            <p style="color:#333; font-size:16px; line-height:1.6;">
                This feature is designed to empower users with insights into efficient nutrient management strategies, improving overall yield outcomes.
            </p>
        </div>
        """

        st.markdown(html_code, unsafe_allow_html=True)

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