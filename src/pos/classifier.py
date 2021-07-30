import json
import os
import sys
import time
import re
from optparse import OptionParser

import matplotlib
import torch

from src.batcher import Batcher
from src.save_utils import save_model, load_model
from models.BERT_RNN import BERT_RNN
from models.RNN_RNN import RNN_RNN
from models.CNN_RNN import CNN_RNN
from models.ATT_RNN import ATT_RNN
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

matplotlib.use('agg')
import matplotlib.pyplot as plt


def _calc_running_avg_loss(loss, running_avg_loss, decay=0.99):
    return decay * running_avg_loss + (1. - decay) * loss if running_avg_loss != 0 else loss


def _train_loop_epoch(model, training_generator, validation_generator):
    training_loss_epoch = []
    validation_loss_epoch = []

    for epoch in range(0, options.epoch):
        print('\tepoch ' + str(epoch))
        loss_train = []
        loss_valid = []

        for batch in training_generator:
            loss, step = model.fit(batch.x, batch.y, batch.msk_conv, batch.msk_msg)
            loss_train.append(loss)
        for batch in validation_generator:
            loss, step = model.valid(batch.x, batch.y, batch.msk_conv, batch.msk_msg)
            loss_valid.append(loss)

        save_model(model, options.output, epoch, options.keep)
        training_loss_epoch.append(sum(loss_train) / len(loss_train))
        validation_loss_epoch.append(sum(loss_valid) / len(loss_valid))

    return training_loss_epoch, validation_loss_epoch


def _train_loop_auto(model, training_generator, validation_generator):
    best_loss = None
    running_avg_loss = 0
    curr_aging = options.age
    training_loss_epoch = []
    validation_loss_epoch = []
    epoch = 0

    while True:
        print('\tepoch ' + str(epoch))
        loss_train = []
        loss_valid = []

        for batch in training_generator:
            loss, step = model.fit(batch.x, batch.y, batch.msk_conv, batch.msk_msg)
            loss_train.append(loss)

        for batch in validation_generator:
            loss, step = model.valid(batch.x, batch.y, batch.msk_conv, batch.msk_msg)
            loss_valid.append(loss)

        avg_valid_loss = sum(loss_valid) / len(loss_valid)
        running_avg_loss = _calc_running_avg_loss(avg_valid_loss, running_avg_loss, options.decay)
        training_loss_epoch.append(sum(loss_train) / len(loss_train))
        validation_loss_epoch.append(avg_valid_loss)

        if (best_loss is None) or running_avg_loss < best_loss:
            print('\tFound a new best loss: {}. Previous best loss: {}'.format(running_avg_loss, best_loss))
            best_loss = running_avg_loss
            save_model(model, options.output, epoch, options.keep)
            curr_aging = options.age
        else:
            curr_aging -= 1
            if curr_aging <= 0:
                break

        epoch += 1

    return training_loss_epoch, validation_loss_epoch


def _compute_metrics(y_test, y_pred, out):
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    out.write('Accuray ' + str(score) + '\n')
    out.write('Precision ' + str(prec) + '\n')
    out.write('Recall ' + str(rec) + '\n')
    out.write('F1 score ' + str(f1) + '\n')
    out.write('\n')
    out.write(str(matrix))
    out.write('\n')


def train_mode(out):
    print('Loading...')

    # Experiment settings
    out.write('*** ' + options.model + ' ***\n\n')
    if options.model not in available_models:
        print('error')
        return

    with open(options.params) as json_file:
        json_params = json.load(json_file)

    params = json_params[options.model]

    # Hyper parameters
    if not re.match(r"\[[\d+.\d+,\s*]*\d+.\d+\]", options.weights):
        print('Invalid weight parameter')
        return sys.exit(22)

    params['batch_size'] = options.batch
    params['msg_len'] = options.msg
    params['conv_len'] = options.conv
    params['path'] = options.output
    params['mode'] = options.mode
    params['bert'] = options.bert
    params['save'] = options.save
    weights = list(map(lambda x: float(x), options.weights[1:-1].split(',')))
    weights = torch.tensor(weights)

    out.write('Hyperparameters: ' + '\n')
    out.write('\tEpoch ' + str(options.epoch) + '\n')
    out.write('\tBatch size ' + str(options.batch) + '\n')
    out.write('\tMax conv length ' + str(options.conv) + '\n')
    out.write('\tMax msg length ' + str(options.msg) + '\n')
    out.write('\t' + str(params) + '\n')
    out.write('\t' + str(weights) + '\n')
    out.flush()

    # Batch generator
    training_generator = Batcher(options.train_data_path, params)
    validation_generator = Batcher(options.valid_data_path, params)

    # Load classifier
    out.write('\nUsing device: ' + str(device) + '\n')
    if device.type == 'cuda':
        out.write(torch.cuda.get_device_name(0) + '\n')

    if options.model == 'BERT_RNN':
        model = BERT_RNN(device, params, weights)
    elif options.model == 'RNN_RNN':
        model = RNN_RNN(device, params, weights)
    elif options.model == 'CNN_RNN':
        model = CNN_RNN(device, params, weights)
    elif options.model == 'ATT_RNN':
        model = ATT_RNN(device, params, weights)
    else:  # default
        print('Model unavailable')
        return sys.exit(22)

    model = model.to(device)

    out.write('\n' + str(model) + '\n')
    out.flush()

    # Training
    print('Training...')
    start_time = time.time()
    if options.epoch == -1:
        training_loss_epoch, validation_loss_epoch = _train_loop_auto(model, training_generator, validation_generator)
    else:
        training_loss_epoch, validation_loss_epoch = _train_loop_epoch(model, training_generator, validation_generator)
    elapsed_time = time.time() - start_time
    out.write('\nTime elapsed ' + str(elapsed_time) + '\n')

    # Plot training and validation loss
    plt.figure()
    plt.plot(training_loss_epoch, label='Training loss')
    plt.plot(validation_loss_epoch, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.legend(loc='upper right')
    plt.draw()
    plt.savefig(options.output + '/loss_per_epoch.png')


def test_mode(out):
    print('Loading...')

    # Hyper parameters
    params = dict()
    params['batch_size'] = 1
    params['msg_len'] = options.msg
    params['conv_len'] = options.conv
    params['mode'] = options.mode
    params['bert'] = options.bert

    # Loading
    out.write('\nUsing device: ' + str(device) + '\n')
    if device.type == 'cuda':
        out.write(torch.cuda.get_device_name(0) + '\n')

    testing_generator = Batcher(options.test, params)

    model = load_model(options.checkpoint)
    model = model.to(device)
    out.write('\n' + str(model) + '\n')
    out.flush()

    # Testing
    print('Testing...')
    y_pred = []
    y_test = []
    for batch in testing_generator:
        pred_y, _ = model.predict(batch.x, batch.msk_conv, batch.msk_msg, no_batch=True)
        true_y = list(filter(lambda z: z != -1, batch.y.view(params['batch_size'] * params['conv_len']).tolist()))

        y_pred.extend(pred_y)
        y_test.extend(true_y)

    _compute_metrics(y_pred, y_test, out)


def demo_mode(out):
    print('Loading...')

    # Hyper parameters
    params = dict()
    params['batch_size'] = 1
    params['msg_len'] = options.msg
    params['conv_len'] = options.conv
    params['mode'] = options.mode
    params['bert'] = options.bert

    # Loading
    testing_generator = Batcher(options.demo, params)

    model = load_model(options.checkpoint)
    model = model.to(device)

    print('Running...')
    predictions = {}
    for batch in testing_generator:
        pred_y, score_y = model.predict(batch.x, batch.msk_conv, batch.msk_msg, no_batch=True)
        true_y = batch.y.tolist()[0]
        predictions[batch.id] = tuple(zip(batch.original, pred_y, score_y, true_y))

    print('Writing results...')
    out.write('id;text;pred;score;gold\n')
    for k, val in predictions.items():
        for msg in val:
            text = []
            tokens = msg[0].split(' ')
            for tok in tokens:
                if tok.startswith('##'):
                    text[-1] = text[-1] + tok[2:]
                else:
                    text.append(tok)

            text = ' '.join(text)
            out.write(k + ';' + text + ';' + str(msg[1]) + ';' + str(msg[2]) + ';' + str(msg[3]) + '\n')


def main():
    if options.mode == 'train':
        with open(options.output + '/out.txt', 'w') as out:
            train_mode(out)
    elif options.mode == 'test':
        with open(options.output + '/test.txt', 'w') as out:
            test_mode(out)
    elif options.mode == 'demo':
        with open(options.output + '/demo.csv', 'w', encoding='utf-8') as out:
            demo_mode(out)
    else:
        print('Error: unknown mode')
        return sys.exit(22)


if __name__ == "__main__":
    print("POS classifier\n")

    argv = sys.argv[1:]
    parser = OptionParser()

    # classifier
    parser.add_option("-m", "--mode", help='mode', action="store", type="string", dest="mode",
                      default="train")
    parser.add_option("-o", "--output", help='output folder', action="store", type="string", dest="output",
                      default="../../output/pos")
    parser.add_option("-t", "--tokenizer", help='bert tokenizer', action="store", type="string", dest="bert",
                      default="dbmdz/bert-base-italian-cased")

    # training
    parser.add_option("--train_data_path", help="path to train dataset", action="store", type="string",
                      default='../../input/pos/train.bin')
    parser.add_option("--valid_data_path", help="path to valid dataset", action="store", type="string",
                      default='../../input/pos/valid.bin')
    parser.add_option("--param", help='parameters', action="store", type="string", dest="params",
                      default="../../input/pos/parameters.json")
    parser.add_option("-c", "--classifier", help='classifier model', action="store", type="string", dest="model",
                      default="RNN_RNN")
    parser.add_option("-e", "--epoch", help='epoch', action="store", type="int", dest="epoch",
                      default=-1)  # -1 for early stopping
    parser.add_option("-a", "--age", help='max age', action="store", type="int", dest="age",
                      default=5)
    parser.add_option("-d", "--decay", help='decay', action="store", type="float", dest="decay",
                      default=0.90)
    parser.add_option("-b", "--batch", help='batch size', action="store", type="int", dest="batch",
                      default=4)
    parser.add_option("-s", "--save", help='save step', action="store", type="int", dest="save",
                      default=100)
    parser.add_option("-w", "--weight", help='loss weights', action="store", type="string", dest="weights",
                      default="[1.86, 2.16]")
    parser.add_option("--keep", help='max keep', action="store", type="int", dest="keep",
                      default=20)
    parser.add_option("--conv", help='conv length', action="store", type="int", dest="conv",
                      default=6)
    parser.add_option("--msg", help='msg length', action="store", type="int", dest="msg",
                      default=20)

    # test and demo
    parser.add_option("--test_data", help='test dataset', action="store", type="string", dest="test",
                      default="../../input/pos/valid.bin")
    parser.add_option("--demo_data", help='demo data folder', action="store", type="string", dest="demo",
                      default="../../input/pos/run.bin")
    parser.add_option("-k", "--checkpoint", help='checkpoint', action="store", type="string", dest="checkpoint",
                      default="../../output/bck/checkpoint/model.pt")

    (options, args) = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    available_models = ['BERT_RNN', 'RNN_RNN', 'CNN_RNN', 'ATT_RNN']

    if options.mode == 'train' and len(os.listdir(options.output)) != 0:
        print('Warning: output directory is not empty, results will be overwritten\n')
        cont = input('Continue (Y|n)? ')
        if cont != 'Y':
            exit(0)

    main()
