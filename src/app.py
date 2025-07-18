from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import sys
import os
import traceback

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__,".."))))

# Import your existing components
from src.pipelines.predict_pipeline import PredictPipeline

app = Flask(__name__)
CORS(app)

# Initialize prediction pipeline
try:
    prediction_pipeline = PredictPipeline()
    print("Prediction pipeline initialized successfully")
except Exception as e:
    print(f"Error initializing prediction pipeline: {e}")
    prediction_pipeline = None

@app.route('/')
def home():
    """Main page with the prediction form"""
    return render_template('index.html')

@app.route('/api')
def api_info():
    """API documentation endpoint"""
    return {
        'message': 'Weather Predictor API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'GET - Web form interface',
            '/health': 'GET - Health check',
            '/predict': 'POST - Weather prediction (JSON)',
            '/predict_form': 'POST - Weather prediction (Form)'
        },
        'required_fields': [
            'Temperature', 'Humidity', 'Wind_Speed', 'Precipitation', 
            'Cloud_cover', 'Atmospheric_pressure', 'UV_Index', 
            'Season', 'Visibility', 'Location'
        ]
    }

@app.route('/health')
def health():
    return {
        'status': 'healthy',
        'pipeline_status': 'ready' if prediction_pipeline else 'error'
    }

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for JSON requests"""
    try:
        if not prediction_pipeline:
            return {
                'error': 'Prediction pipeline not initialized',
                'status': 'error'
            }, 500

        data = request.get_json()
        
        if not data:
            return {
                'error': 'No data provided',
                'status': 'error'
            }, 400

        # Extract required fields
        required_fields = [
            'Temperature', 'Humidity', 'Wind_Speed', 'Precipitation',
            'Cloud_cover', 'Atmospheric_pressure', 'UV_Index', 
            'Season', 'Visibility', 'Location'
        ]
        
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return {
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }, 400

        # Call prediction with all parameters in correct order
        prediction = prediction_pipeline.predict(
            Temperature=data['Temperature'],
            Humidity=data['Humidity'],
            Wind_Speed=data['Wind_Speed'],
            Precipitation=data['Precipitation'],
            Cloud_cover=data['Cloud_cover'],
            Atmospheric_pressure=data['Atmospheric_pressure'],
            UV_Index=data['UV_Index'],
            Season=data['Season'],
            Visibility=data['Visibility'],
            Location=data['Location']
        )
        
        return {
            'prediction': prediction,
            'status': 'success',
            'input_data': data
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'status': 'error',
            'traceback': traceback.format_exc()
        }, 400

@app.route('/predict_form', methods=['POST'])
def predict_form():
    """Form submission endpoint"""
    try:
        if not prediction_pipeline:
            return render_template('result.html', 
                                 error="Prediction pipeline not initialized")

        # Get form data with exact parameter names
        form_data = {
            'Temperature': float(request.form.get('temperature')),
            'Humidity': float(request.form.get('humidity')),
            'Wind_Speed': float(request.form.get('wind_speed')),
            'Precipitation': float(request.form.get('precipitation')),
            'Cloud_cover': request.form.get('cloud_cover'),
            'Atmospheric_pressure': float(request.form.get('atmospheric_pressure')),
            'UV_Index': float(request.form.get('uv_index')),
            'Season': request.form.get('season'),
            'Visibility': float(request.form.get('visibility')),
            'Location': request.form.get('location')
        }

        # Call prediction with exact parameter names and order
        prediction = prediction_pipeline.predict(
            Temperature=form_data['Temperature'],
            Humidity=form_data['Humidity'],
            Wind_Speed=form_data['Wind_Speed'],
            Precipitation=form_data['Precipitation'],
            Cloud_cover=form_data['Cloud_cover'],
            Atmospheric_pressure=form_data['Atmospheric_pressure'],
            UV_Index=form_data['UV_Index'],
            Season=form_data['Season'],
            Visibility=form_data['Visibility'],
            Location=form_data['Location']
        )
        
        return render_template('result.html', 
                             prediction=prediction,
                             input_data=form_data)
        
    except Exception as e:
        return render_template('result.html', 
                             error=str(e),
                             traceback=traceback.format_exc())

@app.errorhandler(404)
def not_found(error):
    return render_template('result.html', error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('result.html', error='Internal server error'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)