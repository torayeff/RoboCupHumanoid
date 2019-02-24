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

DEBUG = False


#
# label::ball | frame1747.jpg | 640 | 480 | 319 | 137 | 409 | 230 | 364.0 | 183.5 | 90 | 93
# [format: "label::annotation_type|filename|img_width|img_height|x1|y1|x2|y2|center_x|center_y|width|height"]
# """
#             The columns in the data are organized as following:
#                 0 -> img_name
#                 1 -> width of the image
#                 2 -> height of the image
#                 3 -> label
#                 4 -> xmin
#                 5 -> ymin
#                 6 -> xmax
#                 7 -> ymax
#         """


def create_csv_from_txt_files(filename, csv_path='', csv_delimiter=';'):
    """Writes a csv file from the text file exported from image tagger

    :param filename: the text file exported form image tagger
           csv_delimiter: csv delimiter, by default ;
            [format: "label::annotation_type|filename|img_width|img_height|x1|y1|x2|y2|center_x|center_y|width|height"]
    """
    with open(filename) as file:
        with open(csv_path + "data.csv", "w") as file_csv:
            csv_header = 'image_file;width;height;label;xmin;ymin;xmax;ymax\n'
            file_csv.write(csv_header)
            csv_rows = ''
            for line in file:

                if line.startswith("label::ball") or line.startswith("ball"):
                    data = line.split('|')
                    csv_rows += data[1] + csv_delimiter  # image name

                    if len(data) > 3:
                        csv_rows += data[2] + csv_delimiter  # img width
                        csv_rows += data[3] + csv_delimiter  # img height
                        csv_rows += data[0].split("::")[1]  # label
                        csv_rows += csv_delimiter + data[4]  # xmin
                        csv_rows += csv_delimiter + data[5]  # ymin
                        csv_rows += csv_delimiter + data[6]  # xmax
                        csv_rows += csv_delimiter + data[7]  # ymax

                    else:
                        csv_rows += '640' + csv_delimiter
                        csv_rows += '480' + csv_delimiter
                        csv_rows += 'ball'

                    csv_rows += '\n'
            file_csv.write(csv_rows)


def get_csv_lines(filename):
    """Creates CSV lines from PASCAL VOC formatted XML file."""
    tree = xml.etree.ElementTree.parse(filename)

    root = tree.getroot()

    size = root.find('size')
    width = size.find('width').text
    height = size.find('height').text
    fn = filename.split("/")[-1].split(".")[0] + ".jpg"
    filedata = [fn, width, height]

    lines = []

    ball_found = False

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        if label == 'ball':
            ball_found = True
            objdata = [label, xmin, ymin, xmax, ymax]
            line = filedata + objdata
            lines.append(line)

    if not ball_found:
        print(filedata)
        lines.append(filedata + ['ball'])

    return lines


def create_csv_folder(xml_folder, csv_folder, sep=','):
    """Creates CSV folder from XML PASCAL VOC folder."""

    strdata = ""
    for fn in glob.glob(xml_folder + "*.xml"):
        lines = get_csv_lines(fn)

        for line in lines:
            strdata += sep.join(line) + "\n"

            # copy to csv_folder
            src = xml_folder + line[0]
            dst = csv_folder + line[0]
            shutil.copy(src, dst)

    with open(csv_folder + "data.csv", "w") as f:
        f.write("image_file,width,height,label,xmin,ymin,xmax,ymax\n")
        f.write(strdata.strip())


def save_pickle(data, output_path='data/teacher_signals'):
    with open(output_path + '.pickle', 'wb') as handler:
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


def detect_peaks(signal, threshold, dist=2):
    """Detect peaks in 2D signal."""

    signal = signal.copy()
    peaks = []

    max_ind = np.unravel_index(np.argmax(signal, axis=None), signal.shape)
    max_val = signal[max_ind]

    while max_val > threshold:
        signal[max_ind] = 0
        peaks.append(max_ind)

        # mask pixels around peak
        for i in range(int(max_ind[0] - dist), int(max_ind[0] + 1 + dist)):
            for j in range(int(max_ind[1] - dist), int(max_ind[1] + 1 + dist)):
                if (i > 0) and (i < signal.shape[0]) and (j > 0) and (j < signal.shape[1]):
                    signal[i, j] = 0

        # determine new maximum value
        max_ind = np.unravel_index(np.argmax(signal, axis=None), signal.shape)
        max_val = signal[max_ind]

    return peaks


def detect_max_peak(signal, threshold, dist=2):
    """Detects only one peak (the max) in 2D signal."""

    signal = signal.copy()
    peaks = []

    max_ind = np.unravel_index(np.argmax(signal, axis=None), signal.shape)
    max_val = signal[max_ind]

    if max_val > threshold:
        signal[max_ind] = 0
        peaks.append(max_ind)

        # mask pixels around peak
        for i in range(int(max_ind[0] - dist), int(max_ind[0] + 1 + dist)):
            for j in range(int(max_ind[1] - dist), int(max_ind[1] + 1 + dist)):
                if (i > 0) and (i < signal.shape[0]) and (j > 0) and (j < signal.shape[1]):
                    signal[i, j] = 0

    return peaks


def evaluate(bndboxes, detections, downsample, radius=5):
    """Calculates the number of true positives, false positives, true negatives and false negatives.

    From paper: A detection is classified as TP if a local maximum with sufficient magnitude is detected within a
    radius of five pixels around the coordinates of the label.

    Note: False positives are calculated if the image does not contain object but output signal contains detections,
        in this case the number of false positives will be the number or detections,
        otherwise the number of false positives is 0.

    Args:
        bndboxes: all bounding boxes around object of ONLY(!) one class in image (not downsampled).
        detections: detection coordinates ONLY(!) for one object.
        downsample: used downsample rate.
        radius: Optional, radius to consider around label coordinates.

    Returns:
        The number of true positives, false positives, true negatives, false negatives.
    """

    centers = []

    # dirty hack to avoid empty boxes
    bndboxes = [box for box in bndboxes if len(box) == 4]

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
            fps = len(detections) - tps  # other detections are false

    return tps, fps, tns, fns


def evaluate_sweaty_model(model, device, dataset, threshold_abs, verbose=False, debug=False):
    """Evaluates given model.

    """
    tic = time()
    print("Evaluating model...")
    # In order to convert tensors to numpy, we need to have those in cpu instead of gpu

    # model is None when debug=True
    if model:
        model.to(device)
        model.eval()

    downsample = dataset[0]['image'].shape[1] / dataset[0]['signal'].shape[1]

    tps = 0
    fps = 0
    tns = 0
    fns = 0

    for i, data in enumerate(dataset):
        if verbose:
            print("Calculating metric for image: {}, [{}/{}]".format(data['img_name'], i, len(dataset)))

        image = data['image'].unsqueeze(0).float().to(device)
        bndboxes = data['bndboxes']
        with torch.no_grad():
            if debug:
                output_signal = np.array(data['signal']).squeeze()  # for debug
                # output_signal = np.zeros(output_signal.shape)  # for debug
            else:
                output_signal = np.array(model(image).squeeze().to(torch.device('cpu')))

            detections = detect_max_peak(output_signal, threshold_abs)

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

    print("Elapsed: {:f} sec.".format(time() - tic))
    print("Results: ", metrics)
    return metrics


def evaluate_sweaty_gru_model(sweaty, conv_gru, device, dataset, threshold_abs, verbose=False, seq_len=15):

    tic = time()
    print("Evaluating model...")

    # model is None when debug=True
    if sweaty and conv_gru:
        sweaty.to(device)
        sweaty.eval()

        conv_gru.to(device)
        conv_gru.eval()

    downsample = dataset[0]['image'].shape[1] / dataset[0]['signal'].shape[1]

    tps = 0
    fps = 0
    tns = 0
    fns = 0

    sequence_input = torch.zeros((seq_len, 1, 120, 160))

    print(sequence_input.size())

    hidden_state = None
    detections = None

    for i, data in enumerate(dataset):

        if verbose:
            print("Calculating metric for image: {}, [{}/{}]".format(data['img_name'], i, len(dataset)))

        image = data['image'].unsqueeze(0).float().to(device)
        bndboxes = data['bndboxes']

        with torch.no_grad():
            sweaty_output = sweaty(image)

            sequence_input = add_sweaty_output_to_seq(sequence_input, sweaty_output, i, i < seq_len)

            if i >= seq_len - 1:
                hidden_state = conv_gru(sequence_input, hidden_state)
                output_to_evaluate = np.array(hidden_state.squeeze().to(torch.device('cpu')))
                detections = detect_max_peak(output_to_evaluate, threshold_abs)

        if detections is not None:
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

    print("Elapsed: {:f} sec.".format(time() - tic))
    print("Results: ", metrics)
    return metrics


def add_sweaty_output_to_seq(seq_input, sweaty_output, index, first_seq):
    if first_seq:
        seq_input[index] = sweaty_output
    else:
        for i in range(0, seq_input.shape[0]-1):
            seq_input[i] = seq_input[i+1]

        seq_input[seq_input.shape[0]-1] = sweaty_output

    return seq_input


class SoccerBallDataset(Dataset):
    """Soccer Balls dataset."""

    def __init__(self, csv_file, root_dir, transform=None, sigma=4, downsample=4,
                 delimiter=";", labels=['ball'], threads=1):
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

                # assert len(data) == 8

                img_name = data[0]
                if img_name not in self.dset.keys():
                    self.dset[img_name] = []
                self.dset[img_name].append(data[1:])

        self.filenames = list(self.dset.keys())

        self.filenames.sort()

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
            if len(box) > 0:  # if we have bndboxes -> means if we have a ball.
                xmin = int(box[0]) // self.downsample
                ymin = int(box[1]) // self.downsample
                xmax = int(box[2]) // self.downsample
                ymax = int(box[3]) // self.downsample

                c_x = (xmax + xmin) / 2
                c_y = (ymax + ymin) / 2

                for y in range(ymin, ymax + 1):
                    for x in range(xmin, xmax + 1):
                        if (y >= 0) and (x >= 0) and (y < s_height) and (x < s_width):
                            signal[0, y, x] += 1000 * scipy.stats.multivariate_normal.pdf([y, x],
                                                                                   [c_y, c_x],
                                                                                   [self.sigma, self.sigma])

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
        if DEBUG:
            print(img_name, self.dset[img_name])

        width = self.dset[img_name][0][0]
        height = self.dset[img_name][0][1]
        bndboxes = []

        # CAUTION ! What if there is no ball in the image? We should clean it
        for obj in self.dset[img_name]:
            if obj[2] in self.labels:  # only get the boundary boxes of the labels we are interested in
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


# create_csv_from_txt_files('data/imageset_432.txt', 'data/imageset_432/')
# create_csv_folder('data/train_cnn/', 'data/lab1/', sep=';')
