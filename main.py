#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def main():
    from train import training_process
    from predict import predict, predict_all_dataset 
    import argparse

    parser = argparse.ArgumentParser(description='main script for training or predicting with model.')

    parser.add_argument('-train', '--train',action='store_true', help='starts the training of the model.')
    parser.add_argument('-cuda', '--cuda', action='store_true', help='use gpu.')
    parser.add_argument('-predict', '--predict',
                        metavar='image_path',
                        help='predicts image from only the given path. example \'../data/images/GICSD_4_1_33.png\' ')
    parser.add_argument('-predict_all', '--predict_all', action='store_true', help='predicts all the dataset and generates a confusion matrix.')
    parser.add_argument('-batch', '--batch',
                        default=32,
                        type=int,
                        metavar='batch_size',
                        help='changeing the batch size')
    parser.add_argument('-epoch', '--epoch',
                        default=100,
                        type=int,
                        metavar='[# epoch]',
                        help='changeing the number of epochs')
    parser.add_argument('-lr', '--lr',
                        default=1e3,
                        type=int,
                        metavar='learning rate',
                        help='changeing the learning rate')
    parser.add_argument('-patience', '--patience',
                        default=3,
                        type=int,
                        metavar='patience',
                        help='changeing the patience')
    parser.add_argument('-worker', '--worker',
                        default=3,
                        type=int,
                        metavar='worker',
                        help='changeing the worker')




    args = parser.parse_args()

    if args.train:
        print('trainig will start')
        training_process(batch_size=args.batch,
                         n_epochs=args.epoch,
                         lr=args.lr,
                         decay=0.1,
                         decay_points=[50],
                         patience=args.patience,
                         worker_n=args.worker,
                         CUDA=args.cuda)
        print('training ended')
    elif args.predict:
        print('predicting the given image')
        predict(path=args.predict, CUDA=args.cuda)

    elif args.predict_all:
        print('predictions for all dataset will be generated.')
        predict_all_dataset(batch_size=args.batch,
                            worker_n=args.worker,
                            CUDA=args.cuda)

if __name__ == '__main__':
    main()

