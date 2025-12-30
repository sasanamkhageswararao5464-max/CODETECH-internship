import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime

# Set your OpenWeatherMap API key
API_KEY = "382ef7a4f0d86e920636973247d5ea32"
CITY = "Visakhapatnam"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Fetch weather data
def fetch_weather_data(city, api_key):
    """
    Fetch current weather data from OpenWeatherMap API
    """
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'  # Use Celsius
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Process and visualize data
def visualize_weather_data(weather_data):
    """
    Create visualizations from weather data
    """
    if not weather_data:
        return

    # Extract relevant data
    city_name = weather_data['name']
    temperature = weather_data['main']['temp']
    feels_like = weather_data['main']['feels_like']
    humidity = weather_data['main']['humidity']
    pressure = weather_data['main']['pressure']
    wind_speed = weather_data['wind']['speed']
    description = weather_data['weather'][0]['description']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Weather Dashboard - {city_name}', fontsize=16, fontweight='bold')

    # 1. Temperature Gauge
    ax1 = axes[0, 0]
    temperatures = [temperature, feels_like]
    labels = ['Current', 'Feels Like']
    colors = ['#FF6B6B', '#FFA07A']
    ax1.bar(labels, temperatures, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Temperature (°C)', fontweight='bold')
    ax1.set_title('Temperature Comparison')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(temperatures):
        ax1.text(i, v + 0.5, f'{v}°C', ha='center', fontweight='bold')

    # 2. Humidity and Pressure
    ax2 = axes[0, 1]
    metrics = ['Humidity', 'Pressure']
    values = [humidity, pressure]
    colors_pie = ['#4ECDC4', '#FFD700']
    ax2.pie(values, labels=metrics, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax2.set_title('Humidity & Pressure Distribution')

    # 3. Wind Speed
    ax3 = axes[1, 0]
    ax3.barh(['Wind Speed'], [wind_speed], color='#95E1D3', edgecolor='black', linewidth=2)
    ax3.set_xlabel('Speed (m/s)', fontweight='bold')
    ax3.set_title('Wind Speed')
    ax3.grid(axis='x', alpha=0.3)
    ax3.text(wind_speed + 0.2, 0, f'{wind_speed} m/s', va='center', fontweight='bold')

    # 4. Weather Description and Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = f"""
    Weather Summary
    ================
    City: {city_name}
    Current Temp: {temperature}°C
    Feels Like: {feels_like}°C
    Humidity: {humidity}%
    Pressure: {pressure} hPa
    Wind Speed: {wind_speed} m/s
    Condition: {description.title()}
    Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('weather_dashboard.png', dpi=300, bbox_inches='tight')
    print("✓ Weather dashboard saved as 'weather_dashboard.png'")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Fetching weather data...")
    weather_data = fetch_weather_data(CITY, API_KEY)

    if weather_data:
        print(f"✓ Successfully fetched data for {weather_data['name']}")
        visualize_weather_data(weather_data)
    else:
        print("✗ Failed to fetch weather data")
