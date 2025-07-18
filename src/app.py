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
        'message': 'Lung Cancer Predictor API',
        'version': '1.0.0',
        'endpoints': {
            '/': 'GET - Web form interface',
            '/health': 'GET - Health check',
            '/predict': 'POST - Lung cancer prediction (JSON)',
            '/predict_form': 'POST - Lung cancer prediction (Form)'
        },
        'required_fields': [
            'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease', 
            'fatigue', 'allergy', 'wheezing', 'alcohol_consuming', 
            'coughing', 'swallowing_difficulty', 'chest_pain'
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

        # Extract required fields for lung cancer prediction
        required_fields = [
            'yellow_fingers', 'anxiety', 'peer_pressure', 'chronic_disease',
            'fatigue', 'allergy', 'wheezing', 'alcohol_consuming',
            'coughing', 'swallowing_difficulty', 'chest_pain'
        ]
        
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return {
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }, 400

        # Call prediction with lung cancer parameters
        prediction = prediction_pipeline.predict(
            yellow_fingers=data['yellow_fingers'],
            anxiety=data['anxiety'],
            peer_pressure=data['peer_pressure'],
            chronic_disease=data['chronic_disease'],
            fatigue=data['fatigue'],
            allergy=data['allergy'],
            wheezing=data['wheezing'],
            alcohol_consuming=data['alcohol_consuming'],
            coughing=data['coughing'],
            swallowing_difficulty=data['swallowing_difficulty'],
            chest_pain=data['chest_pain']
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

        # Get form data for lung cancer prediction
        form_data = {
            'yellow_fingers': request.form.get('yellow_fingers'),
            'anxiety': request.form.get('anxiety'),
            'peer_pressure': request.form.get('peer_pressure'),
            'chronic_disease': request.form.get('chronic_disease'),
            'fatigue': request.form.get('fatigue'),
            'allergy': request.form.get('allergy'),
            'wheezing': request.form.get('wheezing'),
            'alcohol_consuming': request.form.get('alcohol_consuming'),
            'coughing': request.form.get('coughing'),
            'swallowing_difficulty': request.form.get('swallowing_difficulty'),
            'chest_pain': request.form.get('chest_pain')
        }

        # Call prediction with lung cancer parameters
        prediction = prediction_pipeline.predict(
            yellow_fingers=form_data['yellow_fingers'],
            anxiety=form_data['anxiety'],
            peer_pressure=form_data['peer_pressure'],
            chronic_disease=form_data['chronic_disease'],
            fatigue=form_data['fatigue'],
            allergy=form_data['allergy'],
            wheezing=form_data['wheezing'],
            alcohol_consuming=form_data['alcohol_consuming'],
            coughing=form_data['coughing'],
            swallowing_difficulty=form_data['swallowing_difficulty'],
            chest_pain=form_data['chest_pain']
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