# Project PaintingLight V 0.1
# Team of Style2Paints 2020
# Non-commercial usage only


import cv2
import rtree
import scipy
import trimesh
import numpy as np
import tensorflow as tf
from scipy.spatial import ConvexHull
from cv2.ximgproc import createGuidedFilter


assert tf.__version__ == '1.4.0'
assert scipy.__version__ == '1.1.0'
assert trimesh.__version__ == '2.37.1'
assert rtree.__version__ == '0.9.3'


# We use SR-CNN as pre-processing to remove JPEG artifacts in input images.
# You can remove these code if you have high-quality PNG images.
session = tf.Session()
tf.keras.backend.set_session(session)
ip3 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
srcnn = tf.keras.models.load_model('srcnn.net')
srcnn_op = srcnn(tf.pad(ip3 / 255.0, [[0, 0], [16, 16], [16, 16], [0, 0]], 'REFLECT'))[:, 16:-16, 16:-16, :] * 255.0
session.run(tf.global_variables_initializer())
srcnn.load_weights('srcnn.net')


# Global position of light source.
gx = 0.0
gy = 0.0


def run_srcnn(x):
    return session.run(srcnn_op, feed_dict={ip3: x[None, :, :, :]})[0].clip(0, 255).astype(np.uint8)


# Some image resizing tricks.
def min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = min(s1, s0)
    raw_max = min(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


# Some image resizing tricks.
def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


# Some image gradient computing tricks.
def get_image_gradient(dist):
    cols = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]))
    rows = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]))
    return cols, rows


def generate_lighting_effects(stroke_density, content):

    # Computing the coarse lighting effects
    # In original paper we compute the coarse effects using Gaussian filters.
    # Here we use a Gaussian pyramid to get similar results.
    # This pyramid-based result is a bit better than naive filters.
    h512 = content
    h256 = cv2.pyrDown(h512)
    h128 = cv2.pyrDown(h256)
    h64 = cv2.pyrDown(h128)
    h32 = cv2.pyrDown(h64)
    h16 = cv2.pyrDown(h32)
    c512, r512 = get_image_gradient(h512)
    c256, r256 = get_image_gradient(h256)
    c128, r128 = get_image_gradient(h128)
    c64, r64 = get_image_gradient(h64)
    c32, r32 = get_image_gradient(h32)
    c16, r16 = get_image_gradient(h16)
    c = c16
    c = d_resize(cv2.pyrUp(c), c32.shape) * 4.0 + c32
    c = d_resize(cv2.pyrUp(c), c64.shape) * 4.0 + c64
    c = d_resize(cv2.pyrUp(c), c128.shape) * 4.0 + c128
    c = d_resize(cv2.pyrUp(c), c256.shape) * 4.0 + c256
    c = d_resize(cv2.pyrUp(c), c512.shape) * 4.0 + c512
    r = r16
    r = d_resize(cv2.pyrUp(r), r32.shape) * 4.0 + r32
    r = d_resize(cv2.pyrUp(r), r64.shape) * 4.0 + r64
    r = d_resize(cv2.pyrUp(r), r128.shape) * 4.0 + r128
    r = d_resize(cv2.pyrUp(r), r256.shape) * 4.0 + r256
    r = d_resize(cv2.pyrUp(r), r512.shape) * 4.0 + r512
    coarse_effect_cols = c
    coarse_effect_rows = r

    # Normalization
    EPS = 1e-10
    max_effect = np.max((coarse_effect_cols**2 + coarse_effect_rows**2)**0.5)
    coarse_effect_cols = (coarse_effect_cols + EPS) / (max_effect + EPS)
    coarse_effect_rows = (coarse_effect_rows + EPS) / (max_effect + EPS)

    # Refinement
    stroke_density_scaled = (stroke_density.astype(np.float32) / 255.0).clip(0, 1)
    coarse_effect_cols *= (1.0 - stroke_density_scaled ** 2.0 + 1e-10) ** 0.5
    coarse_effect_rows *= (1.0 - stroke_density_scaled ** 2.0 + 1e-10) ** 0.5
    refined_result = np.stack([stroke_density_scaled, coarse_effect_rows, coarse_effect_cols], axis=2)

    return refined_result

def add(f2, f1, top, left):
    #img1 = cv2.imread(f1)
    #img2 = cv2.imread(f2)

    height, width = f1.shape[:2]
    m_h, m_w = f2.shape[:2]
    if(top + height < m_h) and (left + width < m_w):
        f2[top:height + top, left:width + left] = f1
    return f2

def run(image, haikei, mask, ambient_intensity, light_intensity, light_source_height, gamma_correction, stroke_density_clipping, light_color_red, light_color_green, light_color_blue, enabling_multiple_channel_effects):

    # Some pre-processing to resize images and remove input JPEG artifacts.
    raw_image = min_resize(image, 512)
    raw_image = run_srcnn(raw_image)
    raw_image = min_resize(raw_image, 512)
    raw_image = raw_image.astype(np.float32)
    unmasked_image = raw_image.copy()

    if mask is not None:
        alpha = np.mean(d_resize(mask, raw_image.shape).astype(np.float32) / 255.0, axis=2, keepdims=True)
        raw_image = unmasked_image * alpha

    # Compute the convex-hull-like palette.
    h, w, c = raw_image.shape
    flattened_raw_image = raw_image.reshape((h * w, c))
    print(flattened_raw_image)
    raw_image_center = np.mean(flattened_raw_image, axis=0)
    hull = ConvexHull(flattened_raw_image)

    # Estimate the stroke density map.
    intersector = trimesh.Trimesh(faces=hull.simplices, vertices=hull.points).ray
    start = np.tile(raw_image_center[None, :], [h * w, 1])
    direction = flattened_raw_image - start
    print('Begin ray intersecting ...')
    index_tri, index_ray, locations = intersector.intersects_id(start, direction, return_locations=True, multiple_hits=True)
    print('Intersecting finished.')
    intersections = np.zeros(shape=(h * w, c), dtype=np.float32)
    intersection_count = np.zeros(shape=(h * w, 1), dtype=np.float32)
    CI = index_ray.shape[0]
    for c in range(CI):
        i = index_ray[c]
        intersection_count[i] += 1
        intersections[i] += locations[c]
    intersections = (intersections + 1e-10) / (intersection_count + 1e-10)
    intersections = intersections.reshape((h, w, 3))
    intersection_count = intersection_count.reshape((h, w))
    intersections[intersection_count < 1] = raw_image[intersection_count < 1]
    intersection_distance = np.sqrt(np.sum(np.square(intersections - raw_image_center[None, None, :]), axis=2, keepdims=True))
    pixel_distance = np.sqrt(np.sum(np.square(raw_image - raw_image_center[None, None, :]), axis=2, keepdims=True))
    stroke_density = ((1.0 - np.abs(1.0 - pixel_distance / intersection_distance)) * stroke_density_clipping).clip(0, 1) * 255

    # A trick to improve the quality of the stroke density map.
    # It uses guided filter to remove some possible artifacts.
    # You can remove these codes if you like sharper effects.
    guided_filter = createGuidedFilter(pixel_distance.clip(0, 255).astype(np.uint8), 1, 0.01)
    for _ in range(4):
        stroke_density = guided_filter.filter(stroke_density)

    # Visualize the estimated stroke density.
    cv2.imwrite('art1-stroke_density.png', stroke_density.clip(0, 255).astype(np.uint8))
    #stroke_density = cv2.imread('stroke.png')
    # Then generate the lighting effects
    raw_image = unmasked_image.copy()
    lighting_effect = np.stack([
        generate_lighting_effects(stroke_density, raw_image[:, :, 0]),
        generate_lighting_effects(stroke_density, raw_image[:, :, 1]),
        generate_lighting_effects(stroke_density, raw_image[:, :, 2])
    ], axis=2)

    # Using a simple user interface to display results.

    def update_mouse(event, x, y, flags, param):
        global gx
        global gy
        gx = - float(x % w) / float(w) * 2.0 + 1.0
        gy = - float(y % h) / float(h) * 2.0 + 1.0
        if event == cv2.EVENT_LBUTTONUP:
            global sx
            global sy
            sx = x
            sy = y
        return


    light_source_color = np.array([light_color_blue, light_color_green, light_color_red])
    def nothing(x):
        pass

    # Create a black image, a window
    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)
    cv2.createTrackbar('rate', 'image', 0, 90, nothing)
    # create switch for ON/OFF functionality
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)
    cv2.createTrackbar('light-height', 'image', 0, 10, nothing)

    global gx
    global gy
    global sx
    global sy
    sx = 60
    sy = 30
    while True:
        cv2.imshow('image', img)
        light_color_red = cv2.getTrackbarPos('R', 'image') / 255
        light_color_green = cv2.getTrackbarPos('G', 'image') / 255
        light_color_blue = cv2.getTrackbarPos('B', 'image') / 255
        size_r = 0.01 + cv2.getTrackbarPos('rate', 'image') / 100
        light_source_height = cv2.getTrackbarPos('light-height', 'image') / 10
        s = cv2.getTrackbarPos(switch, 'image')

        #if s == 0:
        #    img[:] = 0
        #else:
        #    img[:] = [light_color_blue*255, light_color_green*255, light_color_red*255]

        light_source_color = np.array([light_color_blue, light_color_green, light_color_red])
        light_source_location = np.array([[[light_source_height, gy, gx]]], dtype=np.float32)
        light_source_direction = light_source_location / np.sqrt(np.sum(np.square(light_source_location)))
        final_effect = np.sum(lighting_effect * light_source_direction, axis=3).clip(0, 1)
        if not enabling_multiple_channel_effects:
            final_effect = np.mean(final_effect, axis=2, keepdims=True)
        rendered_image = (ambient_intensity + final_effect * light_intensity) * light_source_color * raw_image
        rendered_image = ((rendered_image / 255.0) ** gamma_correction) * 255.0

        height = rendered_image.shape[0]
        width = rendered_image.shape[1]
        rendered_image2 = cv2.resize(rendered_image , (int(width*size_r), int(height*size_r)))
        mask2 = cv2.resize(mask , (int(width*size_r), int(height*size_r)))

        temp_img = np.zeros((height, width, 3), np.uint8) #黒画像
        IMG1 = add(temp_img,rendered_image2,sy,sx)
        temp_img2 = np.zeros((height, width, 3), np.uint8)  # 黒画像
        IMG1_mask = add(temp_img2, mask2, sy, sx)

        #被写体の周りをぼかすマスク
        temp_img3 = np.zeros((height, width, 3), np.uint8)  # 黒画像
        bokasi_mask = add(temp_img3, 1-mask2, sy, sx)

        haikei = cv2.resize(haikei, rendered_image.shape[1::-1])

        mask1 = IMG1_mask/255
        b_mask1 = bokasi_mask/255
        if s == 0:
            rendered_image = IMG1*mask1 + haikei*(1-mask1)
        else:
            #temp_mask = np.zeros((height, width, 3), np.uint8)
            #cv2.min(1-mask1,b_mask1,temp_mask)
            haikei2 = cv2.GaussianBlur(haikei, (7, 7), 0)
            rendered_image = IMG1 * mask1 + haikei2 * (1-mask1)

        #比較用
        #canvas = np.concatenate([raw_image, rendered_image], axis=1).clip(0, 255).astype(np.uint8)
        #単一結果
        canvas = np.clip(rendered_image,0, 255).astype(np.uint8)
        #print(canvas)
        #haikei = cv2.resize(haikei, canvas.shape[1::-1])
        #mask = cv2.resize(mask, canvas.shape[1::-1])
        #cv2.imshow('',mask)
        #print(mask)
        #mask1 = mask/255
        #print(canvas.shape[1::-1])
        #print(raw_image.shape[1::-1])
        #canvas = canvas*mask1 + haikei*(1-mask1)
        #cv2.imwrite('comp/out2.jpg', dst)
        cv2.imshow('Move your mouse on the canvas to play!',canvas)
        cv2.setMouseCallback('Move your mouse on the canvas to play!', update_mouse)

        cv2.waitKey(10)
