class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'data_wenjian'  # 使用相对路径
        elif dataset == 'mydataset':
            return 'data_magnetic'  # 使用相对路径
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
