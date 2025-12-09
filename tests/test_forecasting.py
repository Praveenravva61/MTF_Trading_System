import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.forecasting import ForecastingAgent

def test_forecasting():
    print("Testing ForecastingAgent...")
    try:
        agent = ForecastingAgent("TCS.NS")
        print("Fetching data...")
        df = agent.fetch_data(period="1y") # Short period for test speed
        print(f"Data fetched: {len(df)} rows")
        
        print("Running pipeline...")
        res = agent.run_pipeline()
        
        print("Metrics:", res['metrics'])
        print("Forecast Price:", res['forecast']['final_price'])
        print("Success!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forecasting()
