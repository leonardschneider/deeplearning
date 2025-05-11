import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"
import "../../../leonardschneider/pickle/pickle"
import "../../../leonardschneider/functor/functor"

-- | Fully connected layer type
--   - [m]: number of input features
--   - [n]: number of output features
--   - t: type of the weights (must be a field)
--   - [p]: total byte size of the weights and biases
type^ dense_layer [m] [n] 't [p] [ts] =
  NN ([m]t) (std_weights [n][m] [n] t) ([n]t)
     ([m]t, [n]t)
     t [] [p] [ts]

-- | Fully connected layer
module dense (R: Real): { type t = R.t
                          val init [s]: (label: [s]u8) -> (m: i64) -> (n: i64)
                                  -> activation_func ([n]t)
                                  -> i32
                                  -> dense_layer [m] [n] t [n * (m * R.sz) + n * R.sz] [n * (m * 1) + n * 1]
                         }  = {

  type t            = R.t

  module lalg   = mk_linalg R
  module util   = utility R
  module w_init = weight_initializer R

  -- | Forward propagation
  -- Applies an affine linear transformation to the incoming data: y = x A^T + b
  let forward (k: i64)
              (m: i64) (n: i64)
              (f: [n]t -> [n]t)
              (_training:bool)
              ((A,b): std_weights [n][m] [n] t)
              (x: [k][m]t)
            : ([k]([m]t, [n]t), [k][n]t) =
    let y = lalg.matmul x (transpose A) |> map (map2 (R.+) b)
    let z = map f y
    in (zip x y, z)

  -- Backward propagation
  let backward (k: i64)
               (m: i64) (n: i64)
               (f': [n]t -> [n]t)
               (_first_layer:bool)
               ((A, _): std_weights [n][m] [n] t)
               (cache: [k]([m]t, [n]t))
               (error: [k][n]t)
             : ([k][m]t, (std_weights [n][m] [n] t)) =
    let (x, y)  = unzip cache
    let y'    : [k][n]t = map f' y
    let delta : [k][n]t = map2 (map2 (R.*)) error y'
    let x'    : [k][m]t = lalg.matmul delta A
    let A_grad: [n][m]t = lalg.matmul (transpose y') x
    let b_grad: [n]t    = map R.sum (transpose y')
    in (x', (A_grad, b_grad))

  module P = pickle

  let init [s] (label: [s]u8) m n (act: activation_func ([n]t)) (seed:i32): dense_layer [m] [n] t [n * (m * R.sz) + n * R.sz] [n * (m * 1) + n * 1] =
    {forward  = \k -> forward k m n act.f,
     backward = \k -> backward k m n act.fd,
     pickle = P.pair (P.array n (P.array m R.pu)) (P.array n R.pu),
     specs = label ++ ".weight" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x02] ++ ((P.pickle P.i64) n) ++ ((P.pickle P.i64) m)
          ++ label ++ ".bias" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x01] ++ ((P.pickle P.i64) n),
     functor = F.(pair (array n (array m scalar)) (array n scalar)),
     w_init = \() -> (w_init.gen_random_array_2d_xavier_uni m n seed, map (\_ -> R.(i32 0)) (0..<n)),
     update_weights = \(w, b) (dw, db) -> (map2 (map2 (R.+)) w dw, map2 (R.+) b db),
    }

}
