import cv2
import numpy as np

haikei = cv2.imread('./photo/red-sunset-at-the-beach.jpg')
comp = cv2.imread('./comp/0001_fg.png')
#mask = None
mask = cv2.imread('./comp/0001_out.png')
comp = cv2.resize(comp, haikei.shape[1::-1])
mask = cv2.resize(mask, haikei.shape[1::-1])
mask1 = mask/255

image = comp*mask1 + haikei*(1-mask1)
#cv2.imwrite('comp/out2.jpg', dst)

ambient_intensity = 0.45
light_intensity = 0.85
light_source_height = 0.1
gamma_correction = 1.0
stroke_density_clipping = 1.2
enabling_multiple_channel_effects = True

light_color_red = 0.6
light_color_green = 0.6
light_color_blue = 0.8

from ProjectPaintingLight6b import run

run(image, haikei, mask, ambient_intensity, light_intensity, light_source_height,
    gamma_correction, stroke_density_clipping, light_color_red, light_color_green,
    light_color_blue, enabling_multiple_channel_effects)

