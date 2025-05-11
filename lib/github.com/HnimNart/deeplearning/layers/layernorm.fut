import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"
import "../../../leonardschneider/pickle/pickle"
import "../../../leonardschneider/functor/functor"
import "../../../diku-dk/statistics/statistics"

-- | Layernorm layer type 
--  - [m]: normalization dimension
--  - [n]: features to normalize over (independent)
type^ layernorm_layer [m] [n] 't [p] [ts] =
  NN ([m][n]t) ([n]t, [n]t) ([m][n]t)
     ([m][n]t, [m][n]t, [n]t, [n]t) ([m][n]t) ([m][n]t)
     t [] [p] [ts]

module pre = {
  module i64 = i64
}

-- | Layernorm layer
module layernorm (R: Real) = {

  type t            = R.t

  module stats = mk_statistics R
  module lalg  = mk_linalg R
  open lalg
  --open R

  -- | Forward propagation
  let forward (k: i64)
              (m: i64)
              (n: i64)
              (eps: f32)
              (_training:bool)
              ((gamma, beta): ([n]t, [n]t))
              (x: [k][m][n]t)
            : ([k]([m][n]t, [m][n]t, [n]t, [n]t), [k][m][n]t) =
    let eps = R.f32 eps
    let X = x |> flatten |> transpose
    let E = map stats.mean X
    let Var = map stats.variance X
    let Y = X |> map3 (\e v x -> map (\x -> (x R.- e) R./ (R.sqrt (v R.+ eps))) x) E Var
    let Z = Y |> map3 (\g b y -> map (\y -> y R.* g R.+ b) y) gamma beta
    let y = Y |> transpose |> unflatten
    let z = Z |> transpose |> unflatten
    let cache = zip4 x y (replicate k E) (replicate k Var)
    in (cache, z)

  -- Backward propagation
  let backward (k: i64)
               (m: i64)
               (n: i64)
               (_first_layer:bool)
               ((gamma, _beta): ([n]t, [n]t))
               (cache: [k]([m][n]t, [m][n]t, [n]t, [n]t))
               (dz: [k][m][n]t)
             : ([k][m][n]t, ([n]t, [n]t)) =
    let (x, y, E, Var) = unzip4 cache
    let E = E[0]
    let Var = Var[0]
    let X = x |> flatten |> transpose
    let Y = y |> flatten |> transpose
    let dZ = dz |> flatten |> transpose
    let dgamma = map2 dotprod dZ Y 
    let dbeta  = map R.sum dZ
    let dY = map2 (\z g -> map (R.* g) z) dZ gamma
    -- check https://neuralthreads.medium.com/layer-normalization-and-how-to-compute-its-jacobian-for-backpropagation-55a549d5936f
    -- for Jacobian calculation
    let J1 = \v -> todiag (replicate (k*m) (R.i64 (k*m))) |> map (map (\x -> (x R.- R.(i32 1)) R./ (R.i64 (k*m)) R./ v))
    let J2 = \x (e: t) v -> let d = map (\x -> x R.- e) x in outer d d |> map (map (R./ (R.i64 (k*m)) R./ (v R.** R.(i32 3))))
    let J  = \x e v -> map2 (map2 (R.-)) (J1 v) (J2 x e v)
    let dX = map4 (\x dy e v -> matvecmul_row (J x e v) dy) X dY E Var
    let dx = dX |> transpose |> unflatten
    in (dx, (dgamma, dbeta))

  module P = pickle

  let init [s] (label: [s]u8) (m: i64) (n: i64) (eps: f32): layernorm_layer [m][n] t [n*R.sz + n*R.sz][n*1+n*1]=
    {forward  = \k -> forward k m n eps,
     backward = \k -> backward k m n,
     pickle = P.(pair (array n R.pu) (array n R.pu)),
     functor = F.(pair (array n scalar) (array n scalar)),
     specs = label ++ ".weight" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x01] ++ ((P.pickle P.i64) n)
          ++ label ++ ".bias"   ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x01] ++ ((P.pickle P.i64) n),
     w_init = \() -> (replicate n (R.i64 1), replicate n (R.i64 0)),
     update_weights = \(w, b) (dw, db) -> (map2 (R.+) w dw, map2 (R.+) b db)
    }

}
