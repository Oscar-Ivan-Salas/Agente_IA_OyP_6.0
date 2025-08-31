#!/usr/bin/env python3
"""
Punto de entrada principal para el Gateway.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws="none"
    )
