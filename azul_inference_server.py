"""
Azul AI Model Inference Server
HTTP REST API serveris AI modeļa inferencei
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import json
from typing import List, Tuple, Dict, Any
import logging

# Konfigurācija
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Spēles konstantes
NUM_COLORS = 5
NUM_FACTORIES = 5
TILES_PER_FACTORY = 4
BOARD_SIZE = 5

app = Flask(__name__)
CORS(app)  # Atļauj CORS visiem endpoint'iem


class DQN(nn.Module):
    """Neironu tīkls (identa ar azul_train.py)"""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 1024)   # 512 → 1024
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)         # 512 → 1024
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)          # 256 → 512
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, action_size)
        self.dropout = nn.Dropout(0.25)          # 0.2 → 0.25 (lielākam modelim)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class AzulInferenceEngine:
    """Inference dzinējs Azul AI modelim"""
    
    def __init__(self, model_path: str = 'azul_best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.state_size = None
        self.action_size = None
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Ielādē iztrenēto modeli"""
        try:
            logger.info(f"Ielādē modeli: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Izvelk parametrus
            if isinstance(checkpoint, dict):
                self.state_size = checkpoint.get('state_size', 100)
                self.action_size = checkpoint.get('action_size', 181)
                
                # Izveido modeli
                self.model = DQN(self.state_size, self.action_size).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Vecāks formāts
                logger.warning("Vecāks modeļa formāts - izmanto default parametrus")
                self.state_size = 100
                self.action_size = 181
                self.model = DQN(self.state_size, self.action_size).to(self.device)
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            logger.info(f"✓ Modelis ielādēts: state_size={self.state_size}, action_size={self.action_size}")
            logger.info(f"✓ Ierīce: {self.device}")
            
        except Exception as e:
            logger.error(f"Kļūda ielādējot modeli: {e}")
            raise
    
    def action_to_index(self, action: Tuple) -> int:
        """Konvertē darbību uz indeksu"""
        color, row, source_idx,action_type  = action
        
        if action_type == 'pass':
            return 180
        elif action_type == 'factory':
            # Factory actions: 0-149 (5 factories × 5 colors × 6 rows)
            return source_idx * 30 + color * 6 + row
        else:  # center
            # Center actions: 150-179 (5 colors × 6 rows)
            return 150 + color * 6 + row
    
    def index_to_action(self, idx: int) -> Tuple:
        """Konvertē indeksu uz darbību"""
        if idx == 180:
            return (0,0,0,'pass')
        elif idx < 150:
            factory_idx = idx // 30
            remainder = idx % 30
            color = remainder // 6
            row = remainder % 6
            return ( color, row,factory_idx,'factory')
        else:
            idx -= 150
            color = idx // 6
            row = idx % 6
            return (  color, row,0,'center')
    
    def predict(self, state: np.ndarray, valid_actions: List[Tuple], 
                epsilon: float = 0.0) -> Tuple:
        """
        Prognozē labāko darbību
        
        Args:
            state: Spēles stāvoklis (numpy array)
            valid_actions: Derīgās darbības
            epsilon: Epsilon greedy parametrs (0.0 = tikai greedy)
        
        Returns:
            Labākā darbība
        """
        if np.random.random() < epsilon:
            return np.random.choice(len(valid_actions))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]
            
            # Maskē nederīgās darbības
            valid_indices = [self.action_to_index(action) for action in valid_actions]
            mask = np.ones(self.action_size) * -np.inf
            mask[valid_indices] = q_values[valid_indices]
            
            best_idx = np.argmax(mask)
            best_action = self.index_to_action(best_idx)
            
            # Atrod darbību sarakstā
            try:
                action_index = valid_actions.index(best_action)
            except ValueError:
                # Ja darbība nav derīga, izvēlas labāko no derīgajām
                logger.warning(f"Predicted action {best_action} not in valid actions")
                action_index = 0
            
            return action_index


# Globālais inference engine
inference_engine = None


@app.route('/health', methods=['GET'])
def health_check():
    """Veselības pārbaude"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference_engine is not None,
        'device': str(inference_engine.device) if inference_engine else None
    })


@app.route('/model/info', methods=['GET'])
def model_info():
    """Atgriež informāciju par modeli"""
    if not inference_engine:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'state_size': inference_engine.state_size,
        'action_size': inference_engine.action_size,
        'device': str(inference_engine.device)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prognozē labāko darbību
    
    Request body:
    {
        "state": [array of floats],
        "valid_actions": [list of action tuples],
        "epsilon": 0.0  (optional)
    }
    
    Response:
    {
        "action_index": int,
        "action": tuple,
        "q_value": float (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validē input
        if 'state' not in data or 'valid_actions' not in data:
            return jsonify({'error': 'Missing required fields: state, valid_actions'}), 400
        
        state = np.array(data['state'], dtype=np.float32)
        valid_json_actions = data['valid_actions']
        valid_actions = []
        for action in valid_json_actions:
            action_tuple:tuple = action['color'],action['row'],action['source_idx'],action['type']
            valid_actions.append(action_tuple)
        #valid_actions = [tuple(action) for action in data['valid_actions']]
        epsilon = data.get('epsilon', 0.0)
        
        # Validē state izmēru
        if len(state) != inference_engine.state_size:
            return jsonify({
                'error': f'Invalid state size. Expected {inference_engine.state_size}, got {len(state)}'
            }), 400
        
        # Validē darbības
        if not valid_actions:
            return jsonify({'error': 'No valid actions provided'}), 400
        
        # Prognozē
        action_index = inference_engine.predict(state, valid_actions, epsilon)
        best_action = valid_actions[action_index]
        
        # Aprēķina Q-value (optional)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(inference_engine.device)
            q_values = inference_engine.model(state_tensor).cpu().numpy()[0]
            action_idx = inference_engine.action_to_index(best_action)
            q_value = float(q_values[action_idx])
        
        return jsonify({
            'action_index': int(action_index),
            'action': list(best_action),
            'q_value': q_value
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Prognozē vairākas darbības vienlaikus
    
    Request body:
    {
        "requests": [
            {
                "state": [...],
                "valid_actions": [...],
                "epsilon": 0.0
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'requests' not in data:
            return jsonify({'error': 'Missing requests field'}), 400
        
        results = []
        for req in data['requests']:
            state = np.array(req['state'], dtype=np.float32)
            valid_actions = [tuple(action) for action in req['valid_actions']]
            epsilon = req.get('epsilon', 0.0)
            
            action_index = inference_engine.predict(state, valid_actions, epsilon)
            best_action = valid_actions[action_index]
            
            results.append({
                'action_index': int(action_index),
                'action': list(best_action)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Pārlādē modeli"""
    try:
        data = request.get_json()
        model_path = data.get('model_path', 'azul_best_model.pth')
        
        global inference_engine
        inference_engine = AzulInferenceEngine(model_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Model reloaded from {model_path}'
        })
        
    except Exception as e:
        logger.error(f"Model reload error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def init_server(model_path: str = 'azul_best_model.pth', 
                host: str = '0.0.0.0', 
                port: int = 5000,
                debug: bool = False):
    """Inicializē un palaiž serveri"""
    global inference_engine
    
    logger.info("="*60)
    logger.info("AZUL AI INFERENCE SERVER")
    logger.info("="*60)
    
    # Ielādē modeli
    try:
        inference_engine = AzulInferenceEngine(model_path)
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        raise
    
    # Palaiž serveri
    logger.info(f"\nServeris startē uz http://{host}:{port}")
    logger.info(f"Pieejamie endpoint'i:")
    logger.info(f"  GET  /health           - Veselības pārbaude")
    logger.info(f"  GET  /model/info       - Modeļa informācija")
    logger.info(f"  POST /predict          - Vienas darbības prognozēšana")
    logger.info(f"  POST /predict/batch    - Vairāku darbību prognozēšana")
    logger.info(f"  POST /model/reload     - Modeļa pārlādēšana")
    logger.info("="*60 + "\n")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Azul AI Inference Server')
    parser.add_argument('--model', type=str, default='azul_best_model.pth',
                       help='Path to the trained model file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=None,
                       help='Port number (default: $PORT env var or 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Railway/Heroku uses PORT environment variable
    port = args.port or int(os.environ.get('PORT', 5000))
    
    init_server(
        model_path=args.model,
        host=args.host,
        port=port,
        debug=args.debug
    )
