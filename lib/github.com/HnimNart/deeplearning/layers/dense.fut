import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"
import "../../../leonardschneider/pickle/pickle"

-- | Fully connected layer type
--   - [m]: number of input features
--   - [n]: number of output features
--   - t: type of the weights (must be a field)
--   - [p]: total byte size of the weights and biases
type^ dense_layer [m] [n] 't [p] =
  NN ([m]t) (std_weights [n][m] [n] t) ([n]t)
     ([m]t, [n]t) ([n]t) ([m]t)
     (apply_grad3 t) [] [p]

-- | Fully connected layer
module dense (R: Real): { type t = R.t
                          val init [s]: (label: [s]u8) -> (m: i64) -> (n: i64)
                                  -> activation_func ([n]t)
                                  -> i32
                                  -> dense_layer [m] [n] t [n * (m * R.sz) + n * R.sz]
                         } = {

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
               (apply_grads: apply_grad3 t)
               ((A,b): std_weights [n][m] [n] t)
               (cache: [k]([m]t, [n]t))
               (error: [k][n]t)
             : ([k][m]t, std_weights [n][m] [n] t) =
    let (x, y)  = unzip cache
    let y'      = map f' y
    let delta   = map2 (map2 (R.*)) error y'
    let A_grad: [n][m]t = lalg.matmul (transpose delta) x
    let b_grad: [n]t    = map (R.sum) (transpose delta)
    let (A', b') = apply_grads n m (A, b) (A_grad, b_grad)

    --- Calc error to backprop to previous layer
    let error' = lalg.matmul delta A
    in (error', (A', b'))

  module P = pickle

  let init [s] (label: [s]u8) m n (act: activation_func ([n]t)) (seed:i32) : dense_layer [m] [n] t [n * (m * R.sz) + n * R.sz] =
    {forward  = \k -> forward k m n act.f,
     backward = \k -> backward k m n act.fd,
     pickle = P.pair (P.array n (P.array m R.pu)) (P.array n R.pu),
     specs = label ++ ".weight" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x02] ++ ((P.pickle P.i64) n) ++ ((P.pickle P.i64) m)
          ++ label ++ ".bias" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x01] ++ ((P.pickle P.i64) n),
     w_init = \() -> (w_init.gen_random_array_2d_xavier_uni m n seed, map (\_ -> R.(i32 0)) (0..<n))}

}
