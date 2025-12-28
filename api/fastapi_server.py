from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import json

class FastAPIServer:
    def __init__(self, config):
        self.config = config
        self.app = FastAPI(title="SmartCity Vision API", version="1.0.0")
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "SmartCity Vision API", "version": "1.0.0"}
        
        @self.app.get("/health")
        async def health():
            return {"status": "healthy"}
    
    def run(self):
        uvicorn.run(
            self.app,
            host=self.config.get('api.host', '0.0.0.0'),
            port=self.config.get('api.port', 8000),
            debug=self.config.get('api.debug', False)
        )