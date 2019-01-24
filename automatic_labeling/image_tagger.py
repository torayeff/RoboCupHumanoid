import sys
import os
import darknet as dn

sys.path.append(os.path.join(os.getcwd(), 'python/'))
dn.set_gpu(0)

net = dn.load_net("cfg/yolov3.cfg".encode('utf-8'), "yolov3.weights".encode('utf-8'), 0)
meta = dn.load_meta("cfg/coco.data".encode('utf-8'))

data_root = "data/imageset_430/"
out = "data/imageset_430_labels.txt"

image_names = sorted(os.listdir(data_root))
n = len(image_names)
result = ""
for i, fname in enumerate(image_names, 1):
    img_file = data_root + fname
    print("Analyzing image[{}/{}]: {}".format(i, n, fname))
    detections = dn.detect(net, meta, img_file.encode("utf-8"))

    in_image = False
    for detection in detections:
        label = detection[0].decode("utf-8")

        if label == "sports ball":

            width = detection[2][2]
            height = detection[2][3]
            c_x = detection[2][0]
            c_y = detection[2][1]
            x1 = int(c_x - (width / 2))
            x2 = int(c_x + (width / 2))
            y1 = int(c_y - (height / 2))
            y2 = int(c_y + (height / 2))

            result += '%s|ball|{"x1":"%d","y1":"%d","x2":"%d","y2":"%d"}|\n'%(fname, x1, y1, x2, y2)
            in_image = True
    if not in_image:
        result += '{}|ball|not in image|\n'.format(fname)


print("Writing results to {}".format(out))
with open(out, "w") as f:
    f.write(result)
print("Success!")

