import json
import os
from pathlib import Path

def update_mcp_config():
    # Ruta al archivo de configuración MCP
    config_path = os.path.expanduser('~/.codeium/windsurf/mcp_config.json')
    backup_path = f"{config_path}.bak"
    
    # Configuración para balldontlie
    balldontlie_config = {
        "enabled": True,
        "url": "https://mcp.balldontlie.io/mcp",
        "transport": "http",
        "headers": {
            "Authorization": "f39b3238-a4a0-4f1e-9314-986b8314b665"
        },
        "description": "API de balldontlie.io para obtener datos de la NBA"
    }
    
    try:
        # Hacer una copia de seguridad del archivo actual
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Crear backup
            with open(backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Se creó una copia de seguridad en: {backup_path}")
            
            # Actualizar la configuración
            if 'mcpServers' not in config:
                config['mcpServers'] = {}
            
            # Agregar o actualizar la configuración de balldontlie
            config['mcpServers']['balldontlie-api'] = balldontlie_config
            
            # Guardar los cambios
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✅ Configuración actualizada exitosamente!")
            
        else:
            # Si el archivo no existe, crearlo
            config = {
                "mcpServers": {
                    "balldontlie-api": balldontlie_config
                }
            }
            
            # Asegurarse de que el directorio existe
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # Guardar la nueva configuración
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print("✅ Archivo de configuración creado exitosamente!")
        
        print("\nConfiguración actual:")
        print(f"- Servidor: balldontlie-api")
        print(f"- Estado: {'HABILITADO' if balldontlie_config['enabled'] else 'DESHABILITADO'}")
        print(f"- URL: {balldontlie_config['url']}")
        
    except Exception as e:
        print(f"❌ Error al actualizar la configuración: {e}")
        if os.path.exists(backup_path):
            print(f"\nSe ha creado una copia de seguridad en: {backup_path}")

if __name__ == "__main__":
    print("=== ACTUALIZACIÓN DE CONFIGURACIÓN MCP ===\n")
    update_mcp_config()
