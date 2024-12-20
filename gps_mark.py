from pyproj import Proj, Transformer
import numpy as np

class GPSMarker:
    '''Classe para obter as coordenadas de um objeto que está sendo detectado, em relação ao barco'''
    def __init__(self, magnetic_declination) -> None:
        self.magnetic_declination = magnetic_declination


    def get_gps_coords(self, gps_txt_path):
        '''Obtém as coordenadas UTM do barco'''
        try:
            with open(gps_txt_path, 'r') as f:
                data = f.readlines()
                x = float(data[0][0:-2])
                y = float(data[1][0:-2])
            return [x, y]
        except (FileNotFoundError, ValueError, IndexError) as e:
            raise ValueError(f"Erro ao processar o arquivo {gps_txt_path}: {e}")
        
        
    def gps_transform(self, utm_coords):
        '''Transforma as coordenadas UTM para WGS84'''
        utm_proj = Proj(proj="utm", zone=33, ellps="WGS84", south=False)
        wgs84_proj = Proj(proj="latlong", datum="WGS84")

        transformer = Transformer.from_proj(utm_proj, wgs84_proj)

        longitude, latitude = transformer.transform(utm_coords[0], utm_coords[1])

        self.boat_coords = [latitude, longitude]
    
        return self.boat_coords
    

    def get_imu_values(self, imu_txt_path):
        '''Obtém os valores do IMU'''
        try:
            with open(imu_txt_path, 'r') as f:
                data = f.readlines()
                x = float(data[0][0:-2])
                y = float(data[1][0:-2])
                z = float(data[2][0:-2])
                self.imu = [x, y, z]
            return self.imu
        except (FileNotFoundError, ValueError, IndexError) as e:
            raise ValueError(f"Erro ao processar o arquivo {imu_txt_path}: {e}")
        

    def angle_object(self, center_ox, center_oy, distance, img):
        '''Calcula o ângulo do objeto em relação ao centro da imagem (barco)'''
        self.distance = distance
        width = img.shape[1]
        height = img.shape[0]

        distance_o2c = np.sqrt((center_ox - height/2)**2 + (center_oy - width/2)**2)
        angle_o2c = np.arctan(distance_o2c/self.distance)

        if center_oy < width/2:
            self.new_angle = -angle_o2c
        elif center_oy > width/2:
            self.new_angle = angle_o2c
        else:
            self.new_angle = 0
        

    def gps_mark(self, gps_txt_path, imu_txt_path):
        '''Calcula as coordenadas do objeto em relação ao barco'''
        gps_coords = self.get_gps_coords(gps_txt_path)
        gps_coords = self.gps_transform(gps_coords)
        imu_values = self.get_imu_values(imu_txt_path)

        magnetic_heading = np.arctan2(imu_values[1], imu_values[0])
        magnetic_heading = np.degrees(magnetic_heading)
        if magnetic_heading < 0:
            magnetic_heading = magnetic_heading + 360
        true_heading = magnetic_heading + self.magnetic_declination + self.new_angle

        theta = np.deg2rad(true_heading)
        delta_x = self.distance * np.cos(theta)
        delta_y = self.distance * np.sin(theta)

        delta_latitude = delta_y / 118320
        delta_longitude = delta_x / (118320 * np.cos(np.deg2rad(gps_coords[1])))
        
        self.obj_longitude = gps_coords[1] + delta_longitude
        self.obj_latitude = gps_coords[0] + delta_latitude
        self.obj_coords = [self.obj_latitude, self.obj_longitude]
      
        return self.obj_coords

