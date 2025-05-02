import "../nn_types"
import "layer_type"
import "../../../leonardschneider/pickle/pickle"

type^ flatten_layer [m][a][b] 't =
    NN ([m][a][b]t) () ([m*a*b]t)
       () ([m*a*b]t) ([m][a][b]t)
       (apply_grad3 t) [] []

module flatten (R:real) : {
  type t = R.t
  val init : (m: i64) -> (a: i64) -> (b: i64)
          -> flatten_layer [m][a][b] t
} = {
  type t = R.t

  let forward [m][a][b] 't
              (k: i64) (_training: bool) () (input: [k][m][a][b]t) : ([k](), [k][m*a*b]t) =
    (replicate k (), map (\image -> flatten_3d image) input)

  let backward (k: i64) (m: i64) (a: i64) (b: i64)
               (_first_layer:bool)
               (_: apply_grad3 t)
               ()
               _
               (error: [k][m*a*b]t) : ([k][m][a][b]t, ()) =
    let error' = map unflatten_3d error
    in (error', ())

  let init m a b : flatten_layer [m][a][b] t =
    {forward  = \k -> forward k,
     backward = \k -> backward k m a b,
     pickle = pickle.cst (),
     specs = [],
     w_init  = \() -> ()}
}
