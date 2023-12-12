import json
import requests


def list_markets():
    url = "https://vega-mainnet-data.commodum.io/api/v2/markets"

    payload = {}
    headers = {"Accept": "application/json"}

    response = requests.request("GET", url, headers=headers, data=payload)

    return {
        edge["node"]["id"]: edge["node"]
        for edge in json.loads(response.text)["markets"]["edges"]
    }


def get_latest_market_data(market_id: str):
    url = f"https://vega-mainnet-data.commodum.io/api/v2/market/data/{market_id}/latest"

    payload = {}
    headers = {"Accept": "application/json"}

    response = requests.request("GET", url, headers=headers, data=payload)

    return json.loads(response.text)["marketData"]
