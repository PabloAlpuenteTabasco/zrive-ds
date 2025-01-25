import pytest
import pandas as pd
import requests
from unittest.mock import Mock, patch
from src.module_1.module_1_meteo_api import (
    get_meteo_data,
    fetch_data_interval,
    calculate_values,
    validate_response
)

# Mock de datos de respuesta de la API
MOCK_RESPONSE = {
    "daily": {
        "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "temperature_2m_mean": [5.0, 6.0, 7.0],
        "precipitation_sum": [0.0, 1.2, 0.8],
        "wind_speed_10m_max": [15.0, 20.0, 18.0],
    }
}

# Datos esperados en formato DataFrame
EXPECTED_DF = pd.DataFrame({
    "city": ["Madrid", "Madrid", "Madrid"],
    "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
    "temperature_2m_mean": [5.0, 6.0, 7.0],
    "precipitation_sum": [0.0, 1.2, 0.8],
    "wind_speed_10m_max": [15.0, 20.0, 18.0]
})


def test_validate_response():
    validate_response(MOCK_RESPONSE)  
    with pytest.raises(RuntimeError, match="La respuesta de la API no contiene la clave 'daily'."):
        validate_response({}) 
    with pytest.raises(RuntimeError, match="Falta la clave esperada 'temperature_2m_mean' en la respuesta de la API."):
        validate_response({"daily": {"date": [], "precipitation_sum": [], "wind_speed_10m_max": []}})

@patch("requests.get")
def test_fetch_data_interval_200(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = MOCK_RESPONSE

    result = fetch_data_interval("Madrid", 40.4168, -3.7038, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))

    pd.testing.assert_frame_equal(pd.DataFrame(result), EXPECTED_DF)

    mock_get.assert_called_once_with(
        "https://archive-api.open-meteo.com/v1/archive?",
        params={
            "latitude": 40.4168,
            "longitude": -3.7038,
            "start_date": "2020-01-01",
            "end_date": "2020-01-03",
            "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
            "timezone": "auto"
        }
    )

@patch("requests.get")

def test_fetch_data_interval_404(mock_get):
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error", response=mock_response)
    mock_get.return_value = mock_response

    with pytest.raises(RuntimeError, match="No se pudo obtener datos después de 5 intentos."):
        fetch_data_interval("Madrid", 40.4168, -3.7038, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))

    assert mock_get.call_count == 5  # Verifica que se intentó 5 veces


def test_calculate_values():
    input_data = EXPECTED_DF.copy()
    input_data["date"] = pd.to_datetime(input_data["date"])
    result = calculate_values(input_data, ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"])

    expected_result = pd.DataFrame({
        "city": ["Madrid"],
        "year": [2020],
        "temperature_2m_mean_mean": [6.0],
        "temperature_2m_mean_max": [7.0],
        "temperature_2m_mean_min": [5.0],
        "precipitation_sum_mean": [0.6666666666666666],
        "precipitation_sum_max": [1.2],
        "precipitation_sum_min": [0.0],
        "wind_speed_10m_max_mean": [17.666666666666668],
        "wind_speed_10m_max_max": [20.0],
        "wind_speed_10m_max_min": [15.0],
    })

    result["year"] = result["year"].astype("int64")
    expected_result["year"] = expected_result["year"].astype("int64")

    pd.testing.assert_frame_equal(result, expected_result)

@patch("src.module_1.module_1_meteo_api.fetch_data_interval")
def test_get_meteo_data(mock_fetch_data_interval):
    mock_fetch_data_interval.return_value = EXPECTED_DF.to_dict("records")

    result = get_meteo_data("Madrid", "2020-01-01", "2020-01-03")
    assert not result.empty
    assert len(result) == 3
    mock_fetch_data_interval.assert_called()