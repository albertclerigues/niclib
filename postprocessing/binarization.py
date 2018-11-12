import copy

import numpy as np
from scipy import ndimage

from niclib.metrics import compute_segmentation_metrics, compute_avg_std_metrics_list

from niclib.io.terminal import printProgressBar

def thresh_size_search(result_set, images, thresholds, lesion_sizes, compute_lesion_metrics=False):
    ground_truth_set = [img.labels[0] for img in images]  # Selecting modality 0
    assert len(result_set) == len(ground_truth_set)

    # Preallocate empty list to store the samples metrics, for each th and ls combination
    metrics_iter = {}
    for thresh in thresholds:
        for min_lesion_size in lesion_sizes:
            metrics_iter["th={}_ls={}".format(thresh, min_lesion_size)] = []

    # Compute and store the results for each sample, thresh and min_lesion_size combination
    print("Evaluating threshold and lesion size for binarization")
    for sample_num, (lesion_probs, true_vol) in enumerate(zip(result_set, ground_truth_set)):
        printProgressBar(sample_num, len(result_set), suffix=" samples processed")

        for thresh in thresholds:
            y_prob = lesion_probs > thresh

            # Get connected components information
            y_prob_labelled, nlesions = ndimage.label(y_prob)

            label_list = np.arange(1, nlesions + 1)
            lesion_volumes = [0]
            if nlesions > 0:
                lesion_volumes = ndimage.labeled_comprehension(y_prob, y_prob_labelled, label_list, np.sum, float, 0)

            for min_lesion_size in lesion_sizes:
                if nlesions > 0:
                    # Set to 0 invalid lesions
                    lesions_to_ignore = [idx + 1 for idx, lesion_vol in enumerate(lesion_volumes) if
                                         lesion_vol < min_lesion_size]

                    rec_vol = copy.deepcopy(y_prob_labelled)
                    rec_vol[np.isin(y_prob_labelled, lesions_to_ignore)] = 0
                else:
                    rec_vol = np.zeros_like(y_prob_labelled)

                metrics_iter["th={}_ls={}".format(thresh, min_lesion_size)].append(
                    compute_segmentation_metrics(true_vol, rec_vol, lesion_metrics=compute_lesion_metrics))
    printProgressBar(len(result_set), len(result_set), suffix=" samples processed")

    # Compute avg_std for each metric th and ls combination
    metrics_list, metrics_names = list(), list()
    for thresh in thresholds:
        for min_lesion_size in lesion_sizes:
            metrics_name = "th={}_ls={}".format(thresh, min_lesion_size)
            metrics_names.append(metrics_name)

            metrics_avg_std = compute_avg_std_metrics_list(metrics_iter[metrics_name])
            metrics_list.append(metrics_avg_std)

    return metrics_list, metrics_names