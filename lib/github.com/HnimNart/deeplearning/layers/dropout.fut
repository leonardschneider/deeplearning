import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../leonardschneider/pickle/pickle"
import "../../../leonardschneider/functor/functor"


-- | Dropout layer type 
type^ dropout_layer [m] 't [p] [ts] =
  NN ([m]t) i32 ([m]t)
     ([m]bool) ([m]t) ([m]t)
     () [] [p] [ts]

-- | Dropout layer
module dropout (R: Real) = {

  type t            = R.t

  module rand = weight_initializer R

  -- | Forward propagation
  -- Drops with probability p a fraction of the input features
  let forward (k: i64)
              (m: i64)
              (p: f32)
              (is_training:bool)
              (seed: i32)
              (x: [k][m]t)
            : ([k][m]bool, [k][m]t) =
    if !is_training then (replicate k (replicate m false), x)
    else
      let scale = 1/(1-p) |> R.f32
      let mask = rand.gen_random_array_uni (k*m) (R.(f32 0), R.(f32 1.0)) seed |> map (R.< R.(f32 p)) |> unflatten
      let y = map2 (map2 (\m x-> if m then R.(i32 0) else (R.*) x scale)) mask x
      in (mask, y)

  -- Backward propagation
  let backward (k: i64)
               (m: i64)
               (p: f32)
               (_first_layer:bool)
               (seed: i32)
               (mask: [k][m]bool)
               (error: [k][m]t)
             : ([k][m]t, i32) =
    let scale = 1/(1-p) |> R.f32
    let grad = map2 (map2 (\m e-> if m then R.(i32 0) else (R.*) e scale)) mask error
    in (grad, seed)

  module P = pickle

  let init [s] m (label: [s]u8) (p: f32) (seed:i32): dropout_layer [m] t [4] [0] =
    {forward  = \k -> forward k m p,
     backward = \k -> backward k m p,
     pickle = P.i32,
     functor = F.cst 0, -- bypass the optimizer
     specs = label ++ ".seed" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x00],
     w_init = \() -> seed,
     update_weights = \seed _ -> seed + 1,
    }

}
