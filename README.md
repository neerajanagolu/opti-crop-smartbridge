# OptiCrop - Smart Agricultural Production Optimization Engine

OptiCrop is an advanced software system that utilizes data-driven insights to optimize agricultural production for different crops. By integrating key environmental factors such as Nitrogen (N), Phosphorous (P), Potassium (K) levels, soil temperature, humidity, pH, and rainfall, it provides intelligent crop recommendations to farmers for maximizing yields.

## How to Run the Application

The source code for the application is located in the **5. Project Development Phase/Code** directory.

1. **Navigate to the Code directory:**
   ```bash
   cd "5. Project Development Phase/Code"
   ```

2. **Install the required dependencies:**
   Make sure you have Python installed. Then, install the packages from the requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Flask server:**
   Run the following command to start the application:
   ```bash
   flask --app api/index.py run
   ```
   *(Alternatively, you can set the `FLASK_APP` environment variable to `api/index.py` and run `flask run`)*

   If the flask related errors came then use this command instead of above command only  after getting error by running above command: python -m flask --app api/index.py run



4. **Access the application:**
   Open your web browser and go to:
   ```
   http://127.0.0.1:5000
   ```

---
*Powered by Machine Learning*