#!/usr/bin/env python3
"""Flask API for species susceptibility predictions."""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.advanced_prediction_pipeline import get_advanced_pipeline
from src.database.species_database import get_species_database
from config.logging_config import get_logger

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)

# built lazily, loading the model takes a few seconds
pipeline = None
species_db = None


def init_pipeline(model_name="SVM"):
    global pipeline, species_db
    if pipeline is None:
        logger.info(f"Initializing advanced pipeline with {model_name} model...")
        pipeline = get_advanced_pipeline(model_name=model_name)
        species_db = pipeline.species_db
        logger.info("Pipeline initialized successfully")
    return pipeline


@app.route('/')
def index():
    return render_template('index_enhanced.html')


@app.route('/api/species/list', methods=['GET'])
def get_all_species():
    try:
        pipeline = init_pipeline()
        species_list = pipeline.get_species_list()

        limit = request.args.get('limit', 1000, type=int)
        offset = request.args.get('offset', 0, type=int)

        paginated = species_list[offset:offset + limit]

        return jsonify({
            'success': True,
            'count': len(paginated),
            'total': len(species_list),
            'species': paginated
        }), 200

    except Exception as e:
        logger.error(f"Error fetching species list: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/species/search', methods=['GET'])
def search_species():
    try:
        query = request.args.get('q', '').strip()

        if not query:
            return jsonify({'error': 'Search query required'}), 400

        pipeline = init_pipeline()
        results = pipeline.search_species(query)

        limit = request.args.get('limit', 50, type=int)
        limited_results = results[:limit]

        return jsonify({
            'success': True,
            'query': query,
            'count': len(limited_results),
            'total_matches': len(results),
            'species': limited_results
        }), 200

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/species/<species_name>', methods=['GET'])
def get_species_info(species_name):
    try:
        pipeline = init_pipeline()

        if pipeline.species_db.species_exists(species_name):
            info = pipeline.species_db.get_species_info(species_name)
            return jsonify({
                'success': True,
                'species': species_name,
                'exists': True,
                'info': info
            }), 200
        else:
            # unknown species, hand back the nearest matches instead
            similar = pipeline.species_db.find_similar_species(species_name, n_neighbors=5)
            return jsonify({
                'success': True,
                'species': species_name,
                'exists': False,
                'similar_species': [s[0] for s in similar],
                'message': f'Species not in database. Showing {len(similar)} most similar species.'
            }), 200

    except Exception as e:
        logger.error(f"Error fetching species info: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/species/<species_name>/similar', methods=['GET'])
def get_similar_species(species_name):
    try:
        pipeline = init_pipeline()
        # cap it, the similarity search is O(n) over the whole table
        n = min(request.args.get('n', 5, type=int), 20)

        similar = pipeline.species_db.find_similar_species(species_name, n_neighbors=n)

        return jsonify({
            'success': True,
            'species': species_name,
            'similar_species': [
                {'name': s[0], 'similarity': float(s[1])}
                for s in similar
            ]
        }), 200

    except Exception as e:
        logger.error(f"Error finding similar species: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """POST {"species": ..., "model": ..., "virus": ...}; unknown species fall
    back to similarity matching."""
    try:
        data = request.get_json()

        if not data or 'species' not in data:
            return jsonify({'error': 'Species name required'}), 400

        species = data.get('species', '').strip()
        model_name = data.get('model', 'SVM')
        virus = data.get('virus', 'BDBV')

        if not species:
            return jsonify({'error': 'Species name cannot be empty'}), 400

        pipeline = init_pipeline(model_name)

        logger.info(f"Predicting for species: {species}")
        result = pipeline.predict_for_species(species, virus)

        return jsonify({
            'success': True,
            'species': species,
            'model': model_name,
            'prediction': 'SUSCEPTIBLE' if result['prediction'] == 1 else 'RESISTANT',
            'probability': float(result['probability']),
            'probability_percent': f"{float(result['probability'])*100:.1f}%",
            'confidence': float(result['confidence']),
            'source': result['source'],
            'data_quality': result['data_quality'],
            'risk_category': get_risk_category(result['probability']),
            'virus': virus,
            'reference_species': result.get('reference_species', [])[:3],
            'metadata': result.get('metadata', {})
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """POST {"species_list": [...], "model": ..., "virus": ...}."""
    try:
        data = request.get_json()

        if not data or 'species_list' not in data:
            return jsonify({'error': 'Species list required'}), 400

        species_list = data.get('species_list', [])
        model_name = data.get('model', 'SVM')
        virus = data.get('virus', 'BDBV')

        if not species_list:
            return jsonify({'error': 'Species list cannot be empty'}), 400

        pipeline = init_pipeline(model_name)

        logger.info(f"Batch predicting for {len(species_list)} species")
        predictions_df = pipeline.predict_batch(species_list, virus)

        results = []
        for _, row in predictions_df.iterrows():
            results.append({
                'species': row['host_species'],
                'prediction': row['prediction'],
                'probability': float(row['probability']) if pd.notna(row['probability']) else None,
                'probability_percent': row['probability_percent'],
                'confidence': float(row['confidence']) if pd.notna(row['confidence']) else 0,
                'risk_category': row['risk_category'],
                'source': row['source'],
                'data_quality': row['data_quality']
            })

        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'model': model_name,
            'virus': virus
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/stats', methods=['GET'])
def get_database_stats():
    try:
        pipeline = init_pipeline()
        stats = pipeline.species_db.get_statistics()

        return jsonify({
            'success': True,
            'statistics': stats
        }), 200

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/high-risk', methods=['GET'])
def get_high_risk_species():
    try:
        pipeline = init_pipeline()
        threshold = request.args.get('threshold', 0.7, type=float)
        limit = request.args.get('limit', 50, type=int)

        high_risk = pipeline.species_db.get_high_risk_species(threshold)[:limit]

        return jsonify({
            'success': True,
            'threshold': threshold,
            'count': len(high_risk),
            'species': [
                {'name': s[0], 'infection_rate': float(s[1])}
                for s in high_risk
            ]
        }), 200

    except Exception as e:
        logger.error(f"Error fetching high-risk species: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    models = ['SVM', 'XGBoost', 'RandomForest', 'LogisticRegression']
    return jsonify({'models': models}), 200


@app.route('/api/status', methods=['GET'])
def status():
    try:
        pipeline = init_pipeline()
        stats = pipeline.species_db.get_statistics()

        return jsonify({
            'status': 'online',
            'project': 'BDBV Host Prediction - Advanced',
            'version': '2.0.0',
            'database_species': stats['total_species'],
            'database_records': stats['total_records']
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


def get_risk_category(probability: float) -> str:
    if probability >= 0.8:
        return "High"
    elif probability >= 0.5:
        return "Medium"
    else:
        return "Low"


if __name__ == '__main__':
    import pandas as pd

    logger.info("="*80)
    logger.info("BDBV HOST PREDICTION - ADVANCED WEB APPLICATION")
    logger.info("="*80)
    logger.info("Starting Flask server...")
    logger.info("Access the app at: http://localhost:5000")
    logger.info("API docs at: http://localhost:5000/api/status")
    logger.info("="*80)

    # warm the pipeline so the first request is not slow
    init_pipeline()

    app.run(debug=True, host='0.0.0.0', port=5000)
