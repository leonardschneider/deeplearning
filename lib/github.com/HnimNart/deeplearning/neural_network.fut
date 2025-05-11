import "../../leonardschneider/pickle/pickle"
import "../../leonardschneider/functor/functor"
import "nn_types"
import "activation_funcs"
import "optimizers/optimizer_type"


module type network = {

  type t
  type pair 'a 'b

  val train [K] 'w 'g 'o 'e2 'i 'state [s] [ps] [ts]:
            NN i w o g o e2 t [s] [ps] [ts] ->
            w ->
            Opt [ts] t state w ->
            state ->
            (lr: f32) ->
            (input: [K]i) ->
            (labels: [K]o) ->
            (batch_sz: i64) ->
            loss_func o t ->
            (state, weights w, t)
  
  --- Combines two networks into one
  val connect_layers 'w1 'w2 'i1 'o1 'o2 'c1 'c2 'e1 'e2 'e22 [s1] [s2] [p1] [p2] [ts1] [ts2]:
                      NN i1 w1 o1 c1 e22 e1 t [s1] [p1] [ts1] ->
                      NN o1 w2 o2 c2 e2 e22 t [s2] [p2] [ts2] ->
                      NN i1 (pair w1 w2) (o2) (pair c1 c2) (e2) (e1) t [s1+s2] [p1+p2] [ts1+ts2]

  val size 'w 'g 'i 'e1 'e2 '^u 'o [s] [p] [ts]: NN i w o g e1 e2 t [s] [p] [ts] -> i64

  --- Initializes the weights of a network
  val init_weights 'w 'g 'i 'e1 'e2 '^u 'o [s] [p] [ts]:
    NN i w o g e1 e2 t [s] [p] [ts] -> weights (*w)

  --- Performs predictions on data set given a network,
  --- input data and activation func
  val predict [K] 'w 'g 'i 'e1 'e2 '^u 'o [s] [p] [ts]: NN i w o g e1 e2 t [s] [p] [ts] ->
                                              weights w ->
                                             [K]i ->
                                             activation_func o ->
                                             [K]o

  --- Calculates the accuracy given a network, input,
  --- labels and activation_func
  val accuracy [K] 'w 'g 'e1 'e2 'i 'o [s] [p] [ts]:
    NN i w o g e1 e2 t [s] [p] [ts] ->
    weights w ->
    [K]i ->
    [K]o ->
    activation_func o ->
    (o -> i32) ->
    t

  --- Calculates the absolute loss given a network, input, labels,
  --- a loss function and classifier aka activation func
  val loss [K] 'w 'g 'e1 'e2 'i 'o [s] [p] [ts]:
    NN i w o g e1 e2 t [s] [p] [ts] ->
    weights w ->
    [K]i ->
    [K]o ->
    loss_func o t ->
    activation_func o ->
    t

  --- activation function wrappers
  val identity : (n: i64) -> activation_func ([n]t)
  val sigmoid  : (n: i64) -> activation_func ([n]t)
  val relu     : (n: i64) -> activation_func ([n]t)
  val tanh     : (n: i64) -> activation_func ([n]t)
  val softmax  : (n: i64) -> activation_func ([n]t)

  --- helper functions for calculating accuracy
  val argmax [n] : [n]t -> i32
  val argmin [n] : [n]t -> i32

}

module neural_network (R:real): network with t = R.t = {

  type t = R.t
  type pair 'a 'b = (a, b)

  module act_funcs = activation_funcs R

  module trainer = trainer R

  let train = trainer.train

  let connect_layers 'w1 'w2 'i1 'o1 'o2 'c1 'c2 'e1 'e2 'e [s1] [s2] [ps1] [ps2] [ts1] [ts2]
                     ({forward=f1, backward=b1,
                        pickle=p1, specs=sp1,
                        functor=fun1, w_init=wi1,
                        update_weights=uw1}: NN i1 w1 o1 c1 e e1 t [s1] [ps1] [ts1])
                     ({forward=f2, backward=b2,
                        pickle=p2, specs=sp2,
                        functor=fun2, w_init=wi2,
                        update_weights=uw2}: NN o1 w2 o2 c2 e2 e t [s2] [ps2] [ts2])
                      : NN i1 (w1,w2) (o2) (c1,c2) (e2) (e1) t [s1+s2] [ps1+ps2] [ts1+ts2]=
    {forward = \k (is_training) (w1, w2) input ->
                 let (c1, res)  = f1 k is_training w1 input
                 let (c2, res2) = f2 k is_training w2 res
                 in (zip c1 c2, res2),
     backward = \k (_) (w1,w2) c (error) ->
                  let (c1,c2) = unzip c
                  let (err2, w2') = b2 k false w2 c2 error
                  let (err1, w1') = b1 k true w1 c1 err2
                  in (err1, (w1', w2')),
     pickle = pickle.pair p1 p2,
     specs = sp1 ++ sp2,
     functor = F.pair fun1 fun2,
     w_init = \() -> (wi1 (), wi2 ()),
     update_weights = \(w1, w2) -> \(dw1, dw2) -> (uw1 w1 dw1, uw2 w2 dw2)
    }

  let size 'w 'g 'i 'e1 'e2 'o [s] [p] [ts] (_: NN i w o g e1 e2 t [s] [p] [ts]): i64 = p

  let init_weights 'w 'g 'i 'e1 'e2 'o [s] [p] [ts]
                   ({forward=_, backward=_, pickle=_, specs=_, functor=_, w_init=w_init}: NN i w o g e1 e2 t [s] [p] [ts]) =
    { weights = w_init () }

  let predict  [K] 'i 'w 'g 'e1 'e2 'o [s] [ps] [ts]
               ({forward=f, backward=_, pickle=_, specs=_, functor=_, w_init=_}: NN i w o g e1 e2 t [s] [ps] [ts])
               (ws: weights w)
               (input: [K]i)
               ({f=class, fd = _}: activation_func o) =
    let ws = get_weights ws
    let (_, output) = f K false ws input
    in map class output


  let accuracy [K] 'w 'g 'e1 'e2 'i 'o [s] [ps] [ts]
               (nn: NN i w o g e1 e2 t [s] [ps] [ts])
               (ws: weights w)
               (input: [K]i)
               (labels: [K]o)
               (classification:activation_func o)
               (f: o -> i32) : t =
    let predictions = predict nn ws input classification |> map f
    let labels      = map f labels
    let total       = map2 (==) predictions labels |> map i32.bool |> i32.sum
    in R.(i32 total)


  let loss [K] 'w 'g 'e1 'e2 'i 'o [s] [ps] [ts]
           (nn:NN i w o g e1 e2 t [s] [ps] [ts])
           (ws: weights w)
           (input:[K]i)
           (labels:[K]o)
           ({f = loss, fd = _}: loss_func o t)
           (classification:activation_func o) =

    let predictions = predict nn ws input classification
    let losses      = map2 loss predictions labels
    in R.sum losses

  --- Breaks if two or more values have max values?
  --- Question is which index should be chosen then?
  let argmax [n] (X:[n]t) : i32 =
    reduce (\n i -> if R.(X[n] > X[i]) then n else i) 0 (map i32.i64 (iota n))

  let argmin [n] (X:[n]t) : i32 =
    reduce (\n i -> if R.(X[n] < X[i]) then n else i) 0 (map i32.i64 (iota n))

  --- activation function wrappers
  let identity = act_funcs.Identity_1d
  let sigmoid  = act_funcs.Sigmoid_1d
  let relu     = act_funcs.Relu_1d
  let tanh     = act_funcs.Tanh_1d
  let softmax  = act_funcs.Softmax_1d

}
