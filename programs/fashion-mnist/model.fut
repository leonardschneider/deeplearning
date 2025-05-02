-- Small example of training a MLP
--
-- Run 'make' to generate data sets.
-- ==
-- compiled input @ batch_16_mnist_100000_f32.bindata
-- compiled input @ batch_32_mnist_100000_f32.bindata
-- compiled input @ batch_64_mnist_100000_f32.bindata
-- compiled input @ batch_128_mnist_100000_f32.bindata

import "../../lib/github.com/HnimNart/deeplearning/deep_learning"
import "../../lib/github.com/HnimNart/deeplearning/util"
import "../../lib/github.com/diku-dk/autodiff/onehot"
import "../../lib/github.com/leonardschneider/pickle/pickle"
import "../../lib/github.com/leonardschneider/futhark-option/option"

module dl = deep_learning f32

let model = \() -> 
  let seed = 1i32

  let l1 = dl.layers.dense "fc1" 784 300 (dl.nn.relu 300) seed
  let l2 = dl.layers.dense "fc2" 300 100 (dl.nn.relu 100) seed
  let l3 = dl.layers.dense "fc3" 100 10 (dl.nn.identity 10) seed

  let nn0 = dl.nn.connect_layers l1 l2
  let nn  = dl.nn.connect_layers nn0 l3
  in nn

--| Entry points

let unpickle 'a (pu: pickle.pu a [])(bs: []u8): a =
  pickle.unpickle pu bs


entry default_weights =
  let nn = model ()
  in dl.nn.init_weights nn

entry load_weights (ws: []u8) =
  let nn = model ()
  in { weights = unpickle nn.pickle ws }

entry save_weights ws: []u8 =
  let nn = model ()
  let ws = get_weights ws
  in pickle.pickle nn.pickle ws

entry specs =
  let nn = model ()
  in nn.specs

--entry predict [K] ws (inputs:[K][784]dl.t) =
--  let nn = model ()
--  in dl.nn.predict nn ws inputs (dl.nn.softmax 10) |> map dl.nn.argmax

let nn = model ()

entry predict = \ws inputs ->
  dl.nn.predict (model ()) ws inputs (dl.nn.softmax 10) |> map dl.nn.argmax
--  let nn = model ()
--  in dl.nn.predict nn ws inputs (dl.nn.softmax 10) |> map dl.nn.argmax


entry train [K]
  ws
  (batch_size: i32)
  (inputs: [K][784]dl.t)
  (labels: [K]u8) =
  let alpha = 0.01
  let nn = model ()
  let encode = onehot.(onehot (arr f32))
  let labels': [K][10]f32 = map (i64.u8 >-> encode) labels
  in dl.train.gradient_descent nn ws alpha
            inputs labels'
            (i64.i32 batch_size) (dl.loss.softmax_cross_entropy_with_logits 10)

entry validate [K] ws (inputs:[K][784]dl.t) (labels: [K]u8) =
  let nn = model ()
  let encode = onehot.(onehot (arr f32))
  let labels': [K][10]f32 = map (i64.u8 >-> encode) labels
  let accuracy = dl.nn.accuracy
    nn ws
    inputs labels'
    (dl.nn.softmax 10) dl.nn.argmax
  let loss = dl.nn.loss
    nn ws
    inputs labels'
    (dl.loss.softmax_cross_entropy_with_logits 10)
    (dl.nn.identity 10)
  in (accuracy, loss)
