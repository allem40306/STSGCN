# -*- coding:utf-8 -*-

import time
import json
import argparse

import numpy as np
import mxnet as mx

from utils import (construct_model, generate_data,
                   masked_mae_np, masked_mape_np, masked_mse_np)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--plot", help="plot network graph", action="store_true")
parser.add_argument("--save", action="store_true", help="save model")
args = parser.parse_args()

config_filename = args.config

with open(config_filename, 'r') as f:
    config = json.loads(f.read())

print(json.dumps(config, sort_keys=True, indent=4))

net = construct_model(config)

batch_size = config['batch_size']
num_of_vertices = config['num_of_vertices']
graph_signal_matrix_filename = config['graph_signal_matrix_filename']
if isinstance(config['ctx'], list):
    ctx = [mx.gpu(i) for i in config['ctx']]
elif isinstance(config['ctx'], int):
    ctx = mx.gpu(config['ctx'])

loaders = []
true_values = []
for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename, num_of_features=num_of_features)):
    if args.test:
        x = x[: 100]
        y = y[: 100]
    y = y[:, :, :, 0:1]
    y = y.squeeze(axis=-1)
    loaders.append(
        mx.io.NDArrayIter(
            x, y if idx == 0 else None,
            batch_size=batch_size,
            shuffle=(idx == 0),
            label_name='label'
        )
    )
    if idx == 0:
        training_samples = x.shape[0]
    else:
        true_values.append(y)

train_loader, val_loader, test_loader = loaders
val_y, test_y = true_values

global_epoch = 1
global_train_steps = training_samples // batch_size + 1
all_info = []
epochs = config['epochs']

sym, arg_params, aux_params = mx.model.load_checkpoint(f"result/{config_filename.split('/')[1]}/STSGCN", epochs)
# sym, arg_params, aux_params = mx.model.load_checkpoint(f'STSGCN_{config_filename.replace("/","_")}', epochs)

mod = mx.mod.Module(
    sym,
    data_names=['data'],
    label_names=['label'],
    context=ctx
)

mod.bind(
    data_shapes=[(
        'data',
        (batch_size, config['points_per_hour'], num_of_vertices, num_of_features)
    ), ],
    label_shapes=[(
        'label',
        (batch_size, config['points_per_hour'], num_of_vertices)
    )]
)

mod.set_params(arg_params, aux_params)


val_loader.reset()
prediction = mod.predict(val_loader)[1].asnumpy()
loss = masked_mae_np(val_y, prediction, 0)
print('loss: %.2f' % (loss), flush=True)

test_loader.reset()
prediction = mod.predict(test_loader)[1].asnumpy()
tmp_info = []
for idx in range(config['num_for_predict']):
    y, x = test_y[:, : idx + 1, :], prediction[:, : idx + 1, :]
    tmp_info.append((
        masked_mae_np(y, x, 0),
        masked_mape_np(y, x, 0),
        masked_mse_np(y, x, 0) ** 0.5
    ))
mae, mape, rmse = tmp_info[-1]
print('MAE: {:.2f}, MAPE: {:.2f}, RMSE: {:.2f}'.format(mae, mape, rmse))
