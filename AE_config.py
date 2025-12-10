# -*- encoding: utf-8 -*-

end_day = '2025-04-20 23:59:59'
BLOCK_SIZE = 2 * 1024 * 1024
GB = 1024 * 1024 * 1024
START_TIME = 3 * 60
image_types = [1, 9, 20, 31, 36]  # top-five image type in traces
meta_file = "VD_creation_meta.csv"
paper_algo_trans = {"lazyload":"Lazyload", "leap": "Leap", "random": "Random", "greedy": "DADI+", "union-min": "VMT-min", "union-avg": "VMT-avg", "upperbound": "Oracle", "topn": "IOCnt", "io_count_time_0.5": "IOCntT", "io_count_time": "IOCntT", "genetic_score": "ThinkAhead", "thinkahead": "ThinkAhead"}
colors = ['#7d4e4e', '#de1f00', '#15D776', '#808080', '#b30086', '#fcbf50', '#011e90', '#837FA3']
bandwidth_colors = ['#7d4e4e', '#de1f00', '#15D776', '#808080', '#b30086', '#fcbf50', '#011e90', '#8A2BE2', "#FFC0CB", 'black', '#00FFFF', '#FF00FF', '#00CED1', '#FF4500', '#8A2BE2', '#DC143C', '#FFD700', '#FF00FF']
algo_color_map = {"lazyload": bandwidth_colors[1], "leap": bandwidth_colors[2], "random": bandwidth_colors[3], "greedy": bandwidth_colors[4], "union": bandwidth_colors[6], "io_count_time": bandwidth_colors[8], "thinkahead": bandwidth_colors[10]}