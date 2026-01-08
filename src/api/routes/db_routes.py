from flask import Blueprint, jsonify, request
from src.ecopackdb.db_connect import get_engine
from sqlalchemy import text
import pandas as pd

db_bp = Blueprint('database', __name__)

@db_bp.route('/tables', methods=['GET'])
def get_tables():
    """Get list of tables in the database"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """))
            tables = [row[0] for row in result.fetchall()]

        return jsonify({
            'status': 'success',
            'tables': tables
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@db_bp.route('/query', methods=['POST'])
def execute_query():
    """Execute a custom SQL query (READ ONLY)"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Query field is required'
            }), 400

        query = data['query'].strip().upper()

        # Safety check - only allow SELECT queries
        if not query.startswith('SELECT'):
            return jsonify({
                'status': 'error',
                'message': 'Only SELECT queries are allowed'
            }), 400

        engine = get_engine()
        df = pd.read_sql(data['query'], engine)

        # Limit results for API response
        preview = df.head(100).to_dict('records')

        return jsonify({
            'status': 'success',
            'data': preview,
            'total_rows': len(df),
            'columns': list(df.columns),
            'limited': len(df) > 100
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@db_bp.route('/table-info/<table_name>', methods=['GET'])
def get_table_info(table_name):
    """Get information about a specific table"""
    try:
        engine = get_engine()

        # Get column information
        with engine.connect() as conn:
            columns_result = conn.execute(text(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table_name}' AND table_schema = 'public'
                ORDER BY ordinal_position;
            """))
            columns = [{'name': row[0], 'type': row[1], 'nullable': row[2]} for row in columns_result.fetchall()]

            # Get row count
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name};"))
            row_count = count_result.fetchone()[0]

        return jsonify({
            'status': 'success',
            'table': table_name,
            'columns': columns,
            'row_count': row_count
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500