import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Configurar logging con formato detallado
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Constantes
COORDINATES = {
    "Madrid": {"latitude": 40.4168, "longitude": -3.7038},
    "London": {"latitude": 51.5074, "longitude": -0.1278},
    "Rio de Janeiro": {"latitude": -22.9068, "longitude": -43.1729},
}

API_URL = "https://archive-api.open-meteo.com/v1/archive?"
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]
MAX_RETRIES = 5
COMBINED_DATA_FILE = "datos_combinados.csv"
OUTPUT_DIR = "graficos"

def get_meteo_data(city: str, start_date: str, end_date: str, interval: str = "month") -> pd.DataFrame:
    """
    Obtiene datos meteorológicos históricos de la API para una ciudad específica en un intervalo mensual.
    Divide las solicitudes en intervalos mensuales para evitar sobrecargar la API.
    """
    if city not in COORDINATES:
        raise ValueError(f"Ciudad no soportada: {city}")
    if interval != "month":
        raise ValueError("El intervalo debe ser 'month'.")

    latitude = COORDINATES[city]["latitude"]
    longitude = COORDINATES[city]["longitude"]
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_data = []

    while current_date <= end_date_dt:
        next_date = (current_date + timedelta(days=32)).replace(day=1)
        interval_end_date = min(next_date - timedelta(days=1), end_date_dt)

        try:
            logging.info(f"Obteniendo datos para {city} del {current_date.strftime('%Y-%m-%d')} al {interval_end_date.strftime('%Y-%m-%d')}...")
            interval_data = fetch_data_interval(city, latitude, longitude, current_date, interval_end_date)
            all_data.extend(interval_data)
        except RuntimeError as e:
            logging.error(e)
        finally:
            current_date = next_date
            time.sleep(1)

    return pd.DataFrame(all_data)

def fetch_data_interval(city: str, latitude: float, longitude: float, start_date: datetime, end_date: datetime) -> list:
    """
    Realiza una solicitud a la API para un intervalo específico de fechas.
    Maneja reintentos en caso de errores de red o respuestas incompletas.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": ",".join(VARIABLES),
        "timezone": "auto"
    }
    retries = 0

    while retries < MAX_RETRIES:
        try:
            response = requests.get(API_URL, params=params)
            response.raise_for_status()
            data = response.json()
            validate_response(data)

            daily_data = data["daily"]
            dates = daily_data["date"]
            return [
                {
                    "city": city,
                    "date": dates[i],
                    "temperature_2m_mean": daily_data["temperature_2m_mean"][i],
                    "precipitation_sum": daily_data["precipitation_sum"][i],
                    "wind_speed_10m_max": daily_data["wind_speed_10m_max"][i],
                }
                for i in range(len(dates))
            ]
        except requests.exceptions.RequestException as e:
            retries += 1
            logging.warning(f"Error en la solicitud: {e}. Reintento {retries}/{MAX_RETRIES}.")
            time.sleep(5)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise
            else:
                logging.error(f"Error HTTP: {e}. Código de estado: {response.status_code}.")
                retries += 1
                time.sleep(5)
    raise RuntimeError(f"No se pudo obtener datos después de {MAX_RETRIES} intentos.")

def validate_response(data: dict):
    """
    Valida que la respuesta de la API contenga las claves necesarias para procesar los datos.
    """
    if "daily" not in data:
        raise RuntimeError("La respuesta de la API no contiene la clave 'daily'.")
    for key in VARIABLES:
        if key not in data["daily"]:
            raise RuntimeError(f"Falta la clave esperada '{key}' en la respuesta de la API.")


def calculate_values(data: pd.DataFrame, variables_metereologicas: list[str]) -> pd.DataFrame:
    data['date'] = pd.to_datetime(data['date'])
    data['year'] = data['date'].dt.year
    resultados = []
    
    ciudades = data.groupby(['city', 'year'])
    for (ciudad, year), group in ciudades:
        resumen = {
            'city': ciudad,
            'year': year
        }
        for variable in variables_metereologicas:
            if variable in group.columns:
                resumen[f'{variable}_mean'] = group[variable].mean()
                resumen[f'{variable}_max'] = group[variable].max()
                resumen[f'{variable}_min'] = group[variable].min()
        
        resultados.append(resumen)
    
    return pd.DataFrame(resultados)

def plot_city_yearly_statistics(data: pd.DataFrame):
    rows = len(VARIABLES)
    fig, axs = plt.subplots(rows, 1, figsize=(10, 6 * rows), sharex=True)

    if rows == 1:
        axs = [axs]

    colors = {
        city: color for city, color in zip(
            data['city'].unique(),
            ['blue', 'green', 'orange']  # Colores para las ciudades
        )
    }

    for i, variable in enumerate(VARIABLES):
        ax = axs[i]
        for city in data['city'].unique():
            city_data = data[data['city'] == city]
            years = city_data['year'].unique()

            ax.plot(city_data['year'], city_data[f'{variable}_mean'], 
                    label=f'{city} Mean', linestyle='-', color=colors[city], marker='o')
            ax.plot(city_data['year'], city_data[f'{variable}_max'], 
                    label=f'{city} Max', linestyle='--', color=colors[city], marker='^')
            ax.plot(city_data['year'], city_data[f'{variable}_min'], 
                    label=f'{city} Min', linestyle=':', color=colors[city], marker='v')

        ax.set_title(f'{variable.capitalize()} (Cities)')
        ax.set_xlabel('Year')
        ax.set_ylabel(variable.capitalize())
        ax.legend()
        ax.grid()

        ax.set_xticks(years)
        ax.set_xticklabels([str(year) for year in years], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig('src/module_1/climate_representation.png', bbox_inches='tight')
    return fig




def main():
    """
    Flujo principal: Carga datos meteorológicos, calcula estadísticas y genera gráficos.
    Si ya existen datos guardados, los reutiliza.
    """
    MODULE_DIR = "src/module_1"
    COMBINED_DATA_FILE = f"{MODULE_DIR}/datos_combinados.csv"

    Path(MODULE_DIR).mkdir(parents=True, exist_ok=True)

    if Path(COMBINED_DATA_FILE).is_file():
        logging.info(f"Cargando datos {COMBINED_DATA_FILE}...")
        combined_data = pd.read_csv(COMBINED_DATA_FILE, parse_dates=["date"])
    else:
        cities = ["Madrid", "London", "Rio de Janeiro"]
        start_date = "2010-01-01"
        end_date = "2020-12-31"
        all_data = []

        for city in cities:
            logging.info(f"Obteniendo datos para {city}...")
            city_data = get_meteo_data(city, start_date, end_date)
            all_data.append(city_data)

        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.to_csv(COMBINED_DATA_FILE, index=False)
        logging.info(f"Datos guardados en {COMBINED_DATA_FILE}")

    calculate_vls = calculate_values(combined_data, VARIABLES)
    plot_city_yearly_statistics(calculate_vls)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
