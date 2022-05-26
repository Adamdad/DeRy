import os

import numpy as np
from mmcls.core.evaluation.eval_metrics import calculate_confusion_matrix
from mmcls.datasets.base_dataset import BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcls.models.losses import accuracy


def make_dataset(image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    for i in range(len(image_ids)):
        item = (os.path.join('data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):
    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


@DATASETS.register_module()
class AIRCRAFT(BaseDataset):
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def load_annotations(self):
        (image_ids, targets, classes, class_to_idx) = find_classes(self.ann_file)
        samples = make_dataset(image_ids, targets)
        self.samples = samples

        data_infos = []
        for filename, gt_label in self.samples:
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)
        return data_infos

    def evaluate(
            self,
            results,
            # gt_labels,
            metric='accuracy',
            metric_options=None,
            logger=None):
        """Evaluate the dataset.
        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['accuracy', 'per_class_acc']
        eval_results = {}
        results = np.vstack(results)
        gt_labels = self.get_gt_labels()
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        #####
        if 'per_class_acc' in metrics:
            confusion_matrix = calculate_confusion_matrix(results, gt_labels).float()
            per_class_acc = (confusion_matrix.diag() /
                             confusion_matrix.sum(dim=1)).mean()
            eval_results['per_class_acc'] = float(per_class_acc.numpy()) * 100

        return eval_results
