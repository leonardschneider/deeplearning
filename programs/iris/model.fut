import "../../lib/github.com/HnimNart/deeplearning/deep_learning"
import "../../lib/github.com/HnimNart/deeplearning/optimizers/sgd"
import "../../lib/github.com/HnimNart/deeplearning/util"
import "../../lib/github.com/diku-dk/autodiff/onehot"
import "../../lib/github.com/leonardschneider/pickle/pickle"

module dl = deep_learning f32

let model = \() -> 
  let seed = 1i32

  let l1 = dl.layers.dense "fc1" 4 5 (dl.nn.relu 5) seed
  let l2 = dl.layers.dense "fc2" 5 3 (dl.nn.identity 3) seed

  let nn = dl.nn.connect_layers l1 l2
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

let nn = model ()

entry predict = \ws inputs ->
  dl.nn.predict (model ()) ws inputs (dl.nn.identity 3)


module optimizer = sgd f32

entry optimizer_init =
  let nn = model ()
  let optim = optimizer.new nn
  in optim.new_state ()

entry train [K]
  (lr: f32)
  ws
  state
  (batch_size: i32)
  (inputs: [K][4]dl.t)
  (labels: [K]u8) =
  let nn = model ()
  let encode = onehot.(onehot (arr f32))
  let labels': [K][3]f32 = map (i64.u8 >-> encode) labels
  let optim = optimizer.new nn
  in dl.nn.train nn ws optim state lr
            inputs labels'
            (i64.i32 batch_size) (dl.loss.softmax_cross_entropy_with_logits 3)

entry validate [K] ws (inputs:[K][4]dl.t) (labels: [K]u8) =
  let nn = model ()
  let encode = onehot.(onehot (arr f32))
  let labels': [K][3]f32 = map (i64.u8 >-> encode) labels
  let accuracy = dl.nn.accuracy
    nn ws
    inputs labels'
    (dl.nn.softmax 3) dl.nn.argmax
  let loss = dl.nn.loss
    nn ws
    inputs labels'
    (dl.loss.softmax_cross_entropy_with_logits 3)
    (dl.nn.identity 3)
  in (accuracy, loss)
