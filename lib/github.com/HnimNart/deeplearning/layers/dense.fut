import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"
import "../../../leonardschneider/pickle/pickle"

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

  -- Forward propagation
  let forward (k: i64)
              (n: i64) (m: i64)
              (act: [n]t -> [n]t)
              (_training:bool)
              ((w,b): std_weights [n][m] [n] t)
              (input: [k][m]t)
            : ([k]([m]t, [n]t), [k][n]t) =
    let res      = lalg.matmul w (transpose input)
    let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
    let res_act  = map (\x -> act x) (res_bias)
    let cache    = zip input res_bias
    in (cache, res_act)

  -- Backward propagation
  let backward (k: i64)
               (n: i64) (m: i64)
               (act: [n]t -> [n]t)
               (_first_layer:bool)
               (apply_grads: apply_grad3 t)
               ((w,b): std_weights [n][m] [n] t)
               (cache: [k]([m]t, [n]t))
               (error: [k][n]t)
             : ([k][m]t, std_weights [n][m] [n] t) =
    let (input, inp_w_bias) = unzip cache
    let deriv    = (map (\x -> act x) inp_w_bias)
    let delta    = transpose (util.hadamard_prod_2d error deriv)
    let w_grad   = lalg.matmul delta input
    let b_grad   = map (R.sum) delta
    let (w', b') = apply_grads n m (w,b) (w_grad, b_grad)

    --- Calc error to backprop to previous layer
    let error' = transpose (lalg.matmul (transpose w) delta)
    in (error', (w', b'))

  module P = pickle

  let init [s] (label: [s]u8) m n (act: activation_func ([n]t)) (seed:i32) : dense_layer [m] [n] t [n * (m * R.sz) + n * R.sz] =
    {forward  = \k -> forward k n m act.f,
     backward = \k -> backward k n m act.fd,
     pickle = P.pair (P.array n (P.array m R.pu)) (P.array n R.pu),
     specs = label ++ ".weight" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x02] ++ ((P.pickle P.i64) n) ++ ((P.pickle P.i64) m)
          ++ label ++ ".bias" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x01] ++ ((P.pickle P.i64) n),
     w_init = \() -> (w_init.gen_random_array_2d_xavier_uni m n seed, map (\_ -> R.(i32 0)) (0..<n))}

}
