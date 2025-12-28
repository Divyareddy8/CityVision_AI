import math
import numpy as np

class GeoUtils:
    def __init__(self):
        pass
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def pixel_to_gps(self, pixel_x, pixel_y, image_size, bounds):
        lat_min, lon_min, lat_max, lon_max = bounds
        
        lat = lat_max - (pixel_y / image_size[1]) * (lat_max - lat_min)
        lon = lon_min + (pixel_x / image_size[0]) * (lon_max - lon_min)
        
        return lat, lon
    
    def gps_to_pixel(self, lat, lon, image_size, bounds):
        lat_min, lon_min, lat_max, lon_max = bounds
        
        pixel_x = int((lon - lon_min) / (lon_max - lon_min) * image_size[0])
        pixel_y = int((lat_max - lat) / (lat_max - lat_min) * image_size[1])
        
        return pixel_x, pixel_y