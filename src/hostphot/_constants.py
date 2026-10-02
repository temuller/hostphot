# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv, find_dotenv
from matplotlib.font_manager import findSystemFonts

# look for a .env file from the current working directory upwards (e.g. for HSC credentials)
load_dotenv(find_dotenv(usecwd=True))
workdir = os.getenv("workdir", "images")

# for plots
font_family = 'serif'
font_families = findSystemFonts(fontpaths=None, fontext='ttf')
for family in font_families:
    if 'P052' in family:
        font_family = "P052"
        break
