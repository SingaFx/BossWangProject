import numpy as np
from PIL import Image

def image_to_numpy_array(paths):
    results = []
    images = []
    w, h = 28, 28
    if len(paths) == 0:
        return [], ""
    for path in paths:
        img = Image.open(path)
        raw_w, raw_h = img.size
        # filter out those don't like characters
        ratio = float(raw_w) / float(raw_h)
        if ratio > 1.0 or ratio < 0.01:
            continue
        if raw_w < raw_h - 1:
            im_expanded = Image.new('L', (raw_h, raw_h), 255)
            x_off = (raw_h - raw_w) / 2
            im_expanded.paste(img, (x_off, 0, x_off + raw_w, raw_h))
        else:
            im_expanded = img
        im_scaled = im_expanded.resize((w, h), Image.ANTIALIAS)
        images.append(np.asarray(im_scaled).reshape(w * h, 1))
        name = path.replace(".jpeg", "_tmp.jpeg")   
        im_scaled.save(name)
        results.append(name)
    
    num_examples = len(images)
    test_pixels = np.zeros((num_examples, w * h), dtype=np.ubyte)
    for i in xrange(0, num_examples):
        test_pixels[i, :] = images[i][:, 0]
    output_file = paths[0][:paths[0].rfind("/")] + "/test-pixels.ubyte"
    np.save(output_file, test_pixels)

    return results, output_file + ".npy"
