from util import *

DATASET_CONF = {
    'dataset_name': os.path.abspath(__file__).split('/')[-2],
    'dataset_path': '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-1]),

    # field conf
    'zone_id': 11,
    'ball_id': 'N',
    'coordx_per_lon': 92106,  # meter
    'coordy_per_lat': 111195,  # meter
    'lon_min': -118.45578327471132,
    'lon_max': -118.43692299300854,
    'lat_min': 34.063397814174095,
    'lat_max': 34.07846466835352,
    'coordx_max': 1737.1451065167744,  # meter
    'coordy_max': 1675.3588504809645,  # meter
    'poi_num': 236,

    # UGV
    'UGV_init_road_pos': {
        'start_road_node_id': 34,
        'end_road_node_id': 35,
        'progress': 0.5,
    },
}
