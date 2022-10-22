from util import *

DATASET_CONF = {
    'dataset_name': os.path.abspath(__file__).split('/')[-2],
    'dataset_path': '/' + os.path.join(*os.path.abspath(__file__).split('/')[:-1]),

    # field conf
    'zone_id': 52,
    'ball_id': 'N',
    'coordx_per_lon': 89540,  # meter
    'coordy_per_lat': 111195,  # meter
    'lon_min': 127.35479052760316,
    'lon_max': 127.37079868566674,
    'lat_min': 36.362931695802445,
    'lat_max': 36.376777883637985,
    'coordx_max': 1433.370473013373,  # meter
    'coordy_max': 1539.6268563728743,  # meter
    'poi_num': 138,

    # UGV conf
    'UGV_init_road_pos': {
        'start_road_node_id': 71,
        'end_road_node_id': 74,
        'progress': 0.5,
    },
}
