from pyproj import Proj, transform

# Projeção UTM Zona 33T
utm_proj = Proj(proj="utm", zone=33, ellps="WGS84", south=False)
wgs84_proj = Proj(proj="latlong", datum="WGS84")

# Coordenadas UTM (Easting, Northing)
utm_coords = (398897.437500, 5044577.000000)

# Converter para latitude e longitude
latitude, longitude = transform(utm_proj, wgs84_proj, utm_coords[0], utm_coords[1])
print(latitude, longitude)