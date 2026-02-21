# ================================================================
# SERVER v3.0 — Multi-Pair WebSocket Server
# ================================================================
# INSTALL: python -m pip install websockets flask flask-cors
# JALANKAN: python server.py
# ================================================================

import asyncio, json, logging, threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import websockets
    HAS_WS = True
except ImportError:
    HAS_WS = False
    print("Install dulu: python -m pip install websockets flask flask-cors")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [SERVER] %(message)s')
log = logging.getLogger()

# State: menyimpan data terbaru semua pair
latest_all = {
    "EURUSD": None, "GBPUSD": None,
    "USDJPY": None, "XAUUSD": None,
    "_macro": {}, "_timestamp": ""
}

connected_clients = set()

# ================================================================
# FLASK
# ================================================================
app = Flask(__name__)
CORS(app)

@app.route('/update_all', methods=['POST'])
def update_all():
    global latest_all
    try:
        data = request.get_json()
        if not data: return jsonify({"status":"error"}), 400
        latest_all.update(data)
        # Broadcast ke semua browser
        asyncio.run(broadcast(json.dumps(latest_all, default=str)))
        pairs_updated = [k for k in data if not k.startswith('_')]
        log.info(f"✅ Update diterima: {', '.join(pairs_updated)}")
        return jsonify({"status":"ok"})
    except Exception as e:
        log.error(f"❌ {e}")
        return jsonify({"status":"error","msg":str(e)}), 500

@app.route('/latest', methods=['GET'])
def get_latest():
    return jsonify(latest_all)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status":"ok","timestamp":datetime.now().isoformat()})

# ================================================================
# WEBSOCKET
# ================================================================
async def ws_handler(websocket, path=None):
    connected_clients.add(websocket)
    log.info(f"🌐 Browser connect! Total: {len(connected_clients)}")
    try:
        await websocket.send(json.dumps(latest_all, default=str))
        async for _ in websocket:
            pass
    except:
        pass
    finally:
        connected_clients.discard(websocket)
        log.info(f"🔌 Browser disconnect. Sisa: {len(connected_clients)}")

async def broadcast(msg):
    if not connected_clients: return
    dead = set()
    for c in connected_clients.copy():
        try: await c.send(msg)
        except: dead.add(c)
    connected_clients -= dead

async def run_ws():
    if not HAS_WS: return
    log.info("🚀 WebSocket: ws://localhost:8765")
    async with websockets.serve(ws_handler, "localhost", 8765):
        await asyncio.Future()

def start_ws():
    if HAS_WS: asyncio.run(run_ws())

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║       🌐 GABUT C4r15                    ║
║  REST : http://localhost:5000           ║
║  WS   : ws://localhost:8765             ║
╚══════════════════════════════════════════╝
""")
    threading.Thread(target=start_ws, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
