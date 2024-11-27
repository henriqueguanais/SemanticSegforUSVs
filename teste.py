import math

# Dados
delta_y = -8.54  # metros (norte-sul)
delta_x = 5.22   # metros (leste-oeste)
lat_deg = 45     # Latitude inicial em graus

# Fatores de conversão
deg_per_meter_lat = 1 / 111320  # Graus por metro para latitude
deg_per_meter_lon = 1 / (111320 * math.cos(math.radians(lat_deg)))  # Graus por metro para longitude

# Deslocamentos
delta_lat = delta_y * deg_per_meter_lat
delta_lon = delta_x * deg_per_meter_lon

print("Deslocamento em latitude (graus):", delta_lat)
print("Deslocamento em longitude (graus):", delta_lon)