from flask import Blueprint, jsonify, send_file
from src.visualization.evaluation_plots import plot_model_comparison
import io
import base64
import matplotlib.pyplot as plt

visualization_bp = Blueprint('visualization', __name__)

@visualization_bp.route('/model-comparison-plot', methods=['GET'])
def model_comparison_plot():
    """Generate model comparison plot"""
    try:
        # This function shows the plot, but for API we need to save it
        # Let's modify to return base64 encoded image
        plot_model_comparison()

        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close()

        # Encode to base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'status': 'success',
            'plot': f'data:image/png;base64,{image_base64}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@visualization_bp.route('/available-plots', methods=['GET'])
def available_plots():
    """Get list of available visualization endpoints"""
    plots = ['model-comparison-plot']
    return jsonify({
        'status': 'success',
        'available_plots': plots
    })