# based on https://stackoverflow.com/a/45639406
import numpy as np
import cv2


def gen_template(img):
    h, w = img.shape[:2]
    x1 = int(w * (1 - h_templ_ratio))
    y1 = int(h * (1 - v_templ_ratio))
    x2 = w
    y2 = h
    return img[y1:y2, x1:x2]


# applies a Canny filter to get the edges
def mat_2_edges(img):
    edged = cv2.Canny(img, 100, 200)
    return edged


def add_black_margins(img, top, bottom, left, right):
    h, w = img.shape[:2]
    result = np.zeros((h + top + bottom, w + left + right, 3), np.uint8)
    result[top : top + h, left : left + w] = img
    return result


# match each input image with its following image (1->2, 2->3)
def match_images(imgs):
    templates_loc = []
    for i in range(0, len(imgs) - 1):
        template = gen_template(imgs[i])
        template = mat_2_edges(template)
        h_templ, w_templ = template.shape[:2]
        # Apply template Matching
        margin_top = margin_bottom = int(h_templ * (1 - v_templ_ratio))
        margin_left = margin_right = int(w_templ * (1 - h_templ_ratio))
        # we need to enlarge the input image prior to call matchTemplate (template needs to be strictly smaller than the input image)
        img = add_black_margins(imgs[i + 1], margin_top, margin_bottom, margin_left, margin_right)
        img = mat_2_edges(img)
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)  # matching function
        # minMaxLoc gets the best match position
        _, _, _, templ_pos = cv2.minMaxLoc(res)
        # as we added margins to the input image we need to subtract the margins width to get the template position relatively to the initial input image (without the black margins)
        rectified_templ_pos = (templ_pos[0] - margin_left, templ_pos[1] - margin_top)
        templates_loc.append(rectified_templ_pos)
        if __name__ == "__main__":
            print("max_loc", rectified_templ_pos)
    return templates_loc


def calc_final_loc(imgs, templates_loc):
    h_final, w_final = 0, imgs[0].shape[1]

    for i, loc in enumerate(templates_loc):
        h = imgs[i].shape[0]
        h2 = imgs[i + 1].shape[0]
        y_templ = int(h * v_templ_ratio)
        h_final += h2 - y_templ - loc[1]
        templates_loc[i] = h_final
    h_final += imgs[0].shape[0]
    templates_loc.insert(0, 0)
    return templates_loc, h_final, w_final


def stitch(imgs, overlap = 0.5):
    global h_templ_ratio, v_templ_ratio
    h_templ_ratio = 1  # horizontal ratio of the input that we will keep to create a template (= 1, we assume width is correct)
    v_templ_ratio = overlap  # vertical ratio of the input that we will keep to create a template (= overlap)

    templates_loc = match_images(imgs)  # templates location

    final_loc, h_final, w_final = calc_final_loc(imgs, templates_loc)
    result = np.zeros((h_final, w_final, 3), np.uint8)
    for img, loc in zip(imgs, final_loc):
        result[loc : loc + img.shape[0]] = img
    return result


if __name__ == "__main__":
    imgs = []
    for i in np.arange(31):
        imgs.append(cv2.imread(f"target/{i}.png"))

    result = stitch(imgs)

    cv2.imwrite("output.png", result)
