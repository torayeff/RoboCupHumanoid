"""
Helper functions.
"""
import os
import glob
import shutil
import numpy as np
import scipy.stats
import xml.etree.ElementTree
from skimage import io
import torch
from torch.utils.data import Dataset
import pickle
from time import time
from multiprocessing import Pool
from skimage.feature.peak import peak_local_max as peak_local_max


DEBUG = False


def get_csv_lines(filename, labels):
    """Creates CSV lines from PASCAL VOC formatted XML file."""
    tree = xml.etree.ElementTree.parse(filename)

    root = tree.getroot()

    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text
    fn = filename.split("/")[-1].split(".")[0] + ".jpg"
    filedata = [fn, width, height]

    lines = []

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        if label in labels:
            objdata = [label, xmin, ymin, xmax, ymax]
            line = filedata + objdata
            lines.append(line)

    return lines


def create_csv_folder(xml_folder, csv_folder, labels, sep=','):
    """Creates CSV folder from XML PASCAL VOC folder."""

    strdata = ""
    for fn in glob.glob(xml_folder + "*.xml"):
        lines = get_csv_lines(fn, labels)

        for line in lines:
            strdata += sep.join(line) + "\n"

            # copy to csv_folder
            src = xml_folder + line[0]
            dst = csv_folder + line[0]
            shutil.copy(src, dst)

    with open(csv_folder + "data.csv", "w") as f:
        f.write("image_file,width,height,label,xmin,ymin,xmax,ymax\n")
        f.write(strdata.strip())


def save_pickle(data, output_path = 'data/teacher_signals'):
    with open(output_path + '.pickle' , 'wb') as handler:
        pickle.dump(data, handler)


def within_radius(c_y, c_x, peak_y, peak_x, radius=5):
    """Determines whether given coordinates are withing given radius."""
    return (abs(c_x - peak_x) <= radius) and (abs(c_y - peak_y) <= radius)


def get_abs_threshold(trainset, p=0.7):
    """Calculates absolute threshold value for the peak detection.

    From paper: Defining the threshold as 70 % of the average magnitude of a maximum over all training data,
    where at least one object of a class is present, has shown to be a good trade-off between achieving a high
    RC and relatively low FDR.

    Args:
        trainset: pytorch trainset object containing training examples.
        p: Optional, percentage of average magnitued.

    Returns:
        Absolute threshold value for peak detection algorithm.

    """
    abs_threshold = 0
    for data in trainset:
        signals = data['signal']
        abs_threshold += torch.max(signals).item()

    abs_threshold /= len(trainset)

    return p * abs_threshold


def evaluate(bndboxes, detections, downsample, radius=5):
    """Calculates the number of true positives, false positives, true negatives and false negatives.

    From paper: A detection is classified as TP if a local maximum with sufficient magnitude is detected within a
    radius of five pixels around the coordinates of the label.

    Note: True negatives are calculated if the image does not contain object but output signal contains detections,
        in this case the number of true negatives will be the number or detections,
        otherwise the number of true negatives is 0.

    Args:
        bndboxes: all bounding boxes around object of ONLY(!) one class in image (not downsampled).
        detections: detection coordinates ONLY(!) for one object.
        downsample: used downsample rate.
        radius: Optional, radius to consider around label coordinates.

    Returns:
        The number of true positives, false positives, true negatives, false negatives.
    """

    centers = []
    for box in bndboxes:
        xmin = int(box[0]) // downsample
        ymin = int(box[1]) // downsample
        xmax = int(box[2]) // downsample
        ymax = int(box[3]) // downsample

        c_x = (xmax + xmin) / 2
        c_y = (ymax + ymin) / 2

        centers.append((c_y, c_x))

    if len(bndboxes) == 0:
        if len(detections) == 0:
            tps = 0
            fps = 0
            tns = 1  # to avoid zero division
            fns = 0
        else:
            tps = 0
            fps = len(detections)  # all detections are false
            tns = 0
            fns = 0
    else:
        if len(detections) == 0:
            tps = 0
            fps = 0
            tns = 0
            fns = len(bndboxes)
        else:
            tps = 0
            tns = 0
            detected = [0] * len(bndboxes)

            for d_y, d_x in detections:
                for i, (c_y, c_x) in enumerate(centers):
                    if within_radius(c_y, c_x, d_y, d_x, radius=radius):
                        tps += 1
                        detected[i] = True
                        break  # it is enough to find one detection

            tps = np.sum(detected)
            fns = len(detected) - tps  # how many missed or did not detect
            fps = len(detections) - tps # not usre

    return tps, fps, tns, fns


def evaluate_model(model, device, trainset, verbose=False):
    """Evaluates given model.

    """
    model.to(device)
    model.eval()

    threshold_abs = get_abs_threshold(trainset, 0.7)

    downsample = trainset[0]['image'].shape[1] / trainset[0]['signal'].shape[1]

    tps = 0
    fps = 0
    tns = 0
    fns = 0

    for i, data in enumerate(trainset):
        if verbose:
            print("Calculating metric for image: {}, [{}/{}]".format(data['img_name'], i, len(trainset)))

        image = data['image'].unsqueeze(0).float().to(device)
        bndboxes = data['bndboxes']
        with torch.no_grad():
            output_signal = np.array(model(image).squeeze())
            # output_signal = np.array(data['signal']).squeeze()
            output_signal = np.zeros(output_signal.shape)
            detections = peak_local_max(output_signal, threshold_abs=threshold_abs, exclude_border=False)

        tp, fp, tn, fn = evaluate(bndboxes, detections, downsample)
        tps += tp
        fps += fp
        tns += tn
        fns += fn

    metrics = {
        'tps': tps,
        'fps': fps,
        'tns': tns,
        'fns': fns
    }

    return metrics


class SoccerBallDataset(Dataset):
    """Soccer Balls dataset."""

    def __init__(self, csv_file, root_dir, transform=None, sigma=4, downsample=4,
                 delimiter=",", labels=['ball'], threads=1):
        """
        Args:
            csv_file: Path to csv file.
            root_dir: Directory with all images.
            transform: Optional transform to be applied.
            sigma: Standard deviation for Gaussian.
            downsample: Downsampling ratio (input output ratio)
            delimiter: Delimeter of the csv file
        """
        self.sigma = sigma
        self.downsample = downsample
        self.labels = labels
        self.threads = threads
        # dset = {img_name : [ [w,h,l1,xmin1,ymin1,xmax1,ymax1], [w,h,l2,xmin2,ymin2,xmax2,ymax2] ] , ...}
        self.dset = {}
        """
            The columns in the data are organized as following: 
                0 -> img_name
                1 -> width of the image
                2 -> height of the image
                3 -> label 
                4 -> xmin
                5 -> ymin
                6 -> xmax
                7 -> ymax
        """
        with open(csv_file, "r") as f:
            next(f)
            for line in f:
                data = line.strip().split(delimiter)
                
                assert len(data) == 8

                img_name = data[0]
                if img_name not in self.dset.keys():
                    self.dset[img_name] = []
                self.dset[img_name].append(data[1:])

        self.filenames = list(self.dset.keys())

        self.teacher_signals = {}

        self.compute_teacher_signals()

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Fetches image with corresponding teacher signal."""

        img_name = self.filenames[idx]
        image_path = os.path.join(self.root_dir, img_name)
        image = io.imread(image_path).transpose(2, 0, 1)  # [channels, height, width]
        image = image / 255.  # normalizing the image
        image = torch.from_numpy(image)

        # transform only images
        if self.transform:
            image = self.transform(image)

        signal, bndboxes = self.teacher_signals[img_name]
          
        sample = {'image': image, 'signal': signal, 'img_name': img_name, 'bndboxes': bndboxes}

        return sample

    def add_teacher_signal(self, img_name):
        """Creates teacher signal for the image.
            Args:
                img_name: name of the image.
            Returns:
                2D feature map of image.
        """
        width, height, bndboxes = self.get_w_h_bnd_from_img(img_name)

        s_height = height // self.downsample
        s_width = width // self.downsample

        signal = np.zeros((1, s_height, s_width))

        for box in bndboxes:
            xmin = int(box[0]) // self.downsample
            ymin = int(box[1]) // self.downsample
            xmax = int(box[2]) // self.downsample
            ymax = int(box[3]) // self.downsample

            c_x = (xmax + xmin) / 2
            c_y = (ymax + ymin) / 2

            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    if (y >= 0) and (x >= 0) and (y < s_height) and (x < s_width):
                        signal[0, y, x] += scipy.stats.multivariate_normal.pdf([y, x],
                                                                               [c_y, c_x],
                                                                               [self.sigma, self.sigma])

        # gives problem with absolute threshold value
        # sg_sum = signal.sum()
        # if sg_sum == 0:
        #     teacher_signal = signal
        # else:
        #     teacher_signal = signal/sg_sum
        teacher_signal = signal

        self.teacher_signals[img_name] = (torch.tensor(teacher_signal), bndboxes)

    def get_w_h_bnd_from_img(self, img_name):
        """
            Args:
                img_name: name of the image.

            Returns:
                height: Height of the image.
                width: Width of the image.
                bndboxes: Coordinates of bounding boxes around objects as a list of tuples.
        """
        width = self.dset[img_name][0][0] 
        height = self.dset[img_name][0][1]
        bndboxes = []

        # CAUTION ! What if there is no ball in the image? We should clean it
        for obj in self.dset[img_name]:
            if obj[2] in self.labels:   # only get the boundary boxes of the labels we are interested in
                bndboxes.append(obj[3:])  # adding the xmin, ymin, xmax, ymax

        if DEBUG and len(bndboxes) == 0:
            print("The image " + img_name + " does not have a ball in it !")

        return int(width), int(height), bndboxes

    def compute_teacher_signals(self):
        """ Precomputes all teacher signals for each image

            Returns:
                Dictionary with key = img_name, value = (teacher_signal, bndboxes)
        """
        tic = time()
        print("Computing teacher signals...")
        if self.threads == 1:
            for img_name in self.filenames:
                self.add_teacher_signal(img_name)
        # TODO: Later if we have time. For now it is not working
        elif self.threads > 1:
            with Pool(self.threads) as pool:
                pool.imap(self.add_teacher_signal, self.filenames)

        print("Elapsed: {:f} sec.".format(time() - tic))


# labels = ['ball']
# create_csv_folder("data/train_cnn/", "data/train/", labels=labels)
