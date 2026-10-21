import os
import requests
import json

# Nuevas credenciales proporcionadas por el usuario
CLIENT_ID = os.environ.get('PARTNER_CLIENT_ID', '')
CLIENT_SECRET = os.environ.get('PARTNER_CLIENT_SECRET', '')

TOKEN_URL = "https://cpservm.com/gateway/token"

def test_auth():
    print(f"--- Probando Autenticación en {TOKEN_URL} ---")
    
    # Payload según estándar OAuth2 (x-www-form-urlencoded)
    # Algunos sistemas requieren grant_type, otros no. Probaremos ambos.
    
    data_variants = [
        # Variante 1: Con grant_type (estándar)
        {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET},
        # Variante 2: Sin grant_type (según ejemplo de docs)
        {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
    ]
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }

    for i, data in enumerate(data_variants, 1):
        print(f"\nIntento {i} (Data: {list(data.keys())}):")
        try:
            # requests.post maneja automáticamente el URL encoding si pasamos 'data'
            response = requests.post(TOKEN_URL, data=data, headers=headers)
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            
            if response.status_code == 200:
                token_data = response.json()
                print("✅ ¡ÉXITO! Token recibido.")
                print(f"Token (truncado): {token_data.get('access_token')[:20]}...")
                return True
        except Exception as e:
            print(f"⚠️ Error en intento {i}: {e}")

    return False

if __name__ == "__main__":
    test_auth()
