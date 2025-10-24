from nba_api.stats.endpoints import scoreboardv2
from datetime import datetime, timedelta
import pandas as pd

def probar_nba_api():
    try:
        # Obtener la fecha de hoy
        hoy = datetime.today()
        fecha_formato_nba = hoy.strftime('%m/%d/%Y')
        
        print(f"Probando NBA-API para la fecha: {fecha_formato_nba}")
        
        # Obtener el scoreboard
        scoreboard = scoreboardv2.ScoreboardV2(game_date=fecha_formato_nba)
        
        # Obtener los dataframes disponibles
        data_frames = scoreboard.get_data_frames()
        
        # Mostrar información sobre los dataframes disponibles
        print("\nDataFrames disponibles:")
        for i, df in enumerate(data_frames):
            if not df.empty:
                print(f"\nDataFrame {i}: {df.shape[0]} filas x {df.shape[1]} columnas")
                print("Primeras columnas:", df.columns.tolist()[:5])
        
        # Si hay un dataframe con información de partidos, mostrarlo
        if data_frames and not data_frames[0].empty:
            print("\nInformación de partidos:")
            partidos = data_frames[0]
            columnas_interesantes = ['GAME_STATUS_TEXT', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 
                                  'GAME_TIME', 'GAME_DATE_EST']
            columnas_disponibles = [col for col in columnas_interesantes if col in partidos.columns]
            
            if columnas_disponibles:
                print(partidos[columnas_disponibles].to_string(index=False))
            else:
                print("No se encontraron las columnas esperadas en los datos.")
        
        # Verificar si hay datos de líneas de partido
        if len(data_frames) > 1 and not data_frames[1].empty:
            print("\nInformación de líneas de partido:")
            print(data_frames[1].head())
        
        return True
        
    except Exception as e:
        print(f"\nError al usar NBA-API: {str(e)}")
        print("Posiblemente no hay partidos programados para hoy o hay un problema de conexión.")
        
        # Intentar con una fecha de temporada regular conocida
        print("\nProbando con una fecha de temporada regular...")
        try:
            fecha_prueba = "12/25/2024"  # Día de Navidad, típicamente con partidos
            print(f"\nProbando con fecha: {fecha_prueba}")
            scoreboard = scoreboardv2.ScoreboardV2(game_date=fecha_prueba)
            data_frames = scoreboard.get_data_frames()
            
            if data_frames and not data_frames[0].empty:
                print("\n¡Éxito! Se encontraron partidos para la fecha de prueba.")
                partidos = data_frames[0]
                columnas_interesantes = ['GAME_STATUS_TEXT', 'HOME_TEAM_NAME', 'VISITOR_TEAM_NAME', 
                                      'GAME_TIME', 'GAME_DATE_EST']
                columnas_disponibles = [col for col in columnas_interesantes if col in partidos.columns]
                
                if columnas_disponibles:
                    print(partidos[columnas_interesantes].to_string(index=False))
                return True
            else:
                print("No se encontraron partidos ni siquiera en la fecha de prueba.")
                return False
                
        except Exception as e2:
            print(f"Error en la prueba con fecha fija: {str(e2)}")
            return False

# Ejecutar la prueba
if __name__ == "__main__":
    print("=== PRUEBA DE NBA-API ===")
    if probar_nba_api():
        print("\n✅ Prueba completada con éxito.")
    else:
        print("\n❌ Hubo problemas al ejecutar la prueba.")
