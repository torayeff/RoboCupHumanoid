import sys
import os
import darknet as dn

sys.path.append(os.path.join(os.getcwd(), 'python/'))
dn.set_gpu(1)

net = dn.load_net("cfg/yolov3.cfg".encode('utf-8'), "yolov3.weights".encode('utf-8'), 0)
meta = dn.load_meta("cfg/coco.data".encode('utf-8'))
r = dn.detect(net, meta, "/Users/torayeff/darknet/data/dog.jpg".encode("utf-8"))
print(r[:10])


# And then down here you could detect a lot more images like:
# r = dn.detect(net, meta, "/Users/torayeff/darknet/data/giraffe.jpg".encode("utf-8"))
# print(r)
# r = dn.detect(net, meta, "/Users/torayeff/darknet/data/horses.jpg".encode("utf-8"))
# print(r)
# r = dn.detect(net, meta, "/Users/torayeff/darknet/data/person.jpg".encode("utf-8"))
# print(r)

