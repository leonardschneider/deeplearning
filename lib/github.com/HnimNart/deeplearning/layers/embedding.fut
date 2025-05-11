import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../leonardschneider/pickle/pickle"
import "../../../leonardschneider/functor/functor"


-- | Embedding layer type
--  - [m]: number of input tokens
--  - [v]: number of tokens in the vocabulary
--  - [d]: embedding dimension
type^ embedding_layer [m][d][v] 't [p] [ts] =
  NN ([m]u32) ([v][d]t) ([m][d]t)
     ([m]u32)
     t [] [p] [ts]

-- | Embedding layer
module embedding (R: Real) = {

  type t = R.t

  -- | Forward propagation
  -- Drops with probability p a fraction of the input features
  let forward (k: i64)
              (m: i64)
              (v: i64)
              (d: i64)
              (_: bool)
              (w: [v][d]t)
              (x: [k][m]u32)
            : ([k][m]u32, [k][m][d]t) =
    let y = map (map (\i -> w[i64.u32 i])) x
    in (x, y)

  -- Backward propagation
  let backward (k: i64)
               (m: i64)
               (v: i64)
               (d: i64)
               (padding_idx: i64)
               (_first_layer:bool)
               (_w: [v][d]t)
               (x: [k][m]u32)
               (error: [k][m][d]t)
             : ([k][m]u32, [v][d]t) =
    let grads = replicate v (replicate d (R.(i32 0)))
    let x' = replicate k (replicate m 0)
    let is = x |> flatten |> map (\i -> i64.u32 i) |> map (\i -> iota d |> map (\j -> (i, j))) |> flatten
    let grads = reduce_by_index_2d grads (R.+) R.(i32 0) is (flatten_3d error)
    let grads = grads with [padding_idx] = replicate d (R.(i32 0))
    in (x', grads)

  module P = pickle
  module w_init = weight_initializer R

  let init [s] m (label: [s]u8) (d: i64) (v: i64) (padding_idx: i64) (seed:i32): embedding_layer [m][d][v] t [v * (d * R.sz)] [v * (d * 1)] =
    {forward  = \k -> forward k m v d,
     backward = \k -> backward k m v d padding_idx,
     pickle = P.(array v (array d R.pu)),
     functor = F.(array v (array d scalar)),
     specs = label ++ ".weight" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x02] ++ ((P.pickle P.i64) v) ++ ((P.pickle P.i64) d),
     w_init = \() -> w_init.gen_random_array_norm (v*d) seed { mean = R.(f32 0), stddev = R.(f32 1.0) } |> unflatten,
     update_weights = \w dw -> map2 (map2 (R.+)) w dw,
    }

}
