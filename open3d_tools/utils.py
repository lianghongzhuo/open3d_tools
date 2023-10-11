import os
import json

dir_name = os.path.dirname(os.path.abspath(__file__))
with open(f"{dir_name}/../config/m_colors.json") as json_file:
    m_colors = json.load(json_file)


def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip("#")
    hex_len = len(hex_value)
    return tuple(int(hex_value[i:i+hex_len//3], 16) / 255.0 for i in range(0, hex_len, hex_len//3))


def get_rgb_colors():
    rgb_colors = {}
    for key in m_colors.keys():
        if isinstance(m_colors[key], str):
            color = hex_to_rgb(m_colors[key])
            rgb_colors[key] = color
    return rgb_colors
