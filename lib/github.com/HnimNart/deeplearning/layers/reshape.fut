import "../nn_types"
import "layer_type"
import "../../../leonardschneider/pickle/pickle"
import "../../../leonardschneider/functor/functor"

type^ reshape_layer 'a 'b 't =
    NN a () b
       () b a
       t [] [] []

module reshape (R:real) = {
  type t = R.t

  let forward 'a 'b
              (k: i64)
              (f: a -> b)
              (_training: bool)
              ()
              (input: [k]a)
              : ([k](), [k]b) =
    (replicate k (), map f input)

  let backward 'a 'b
              (k: i64)
              (g: b -> a)
              (_first_layer:bool)
              _w
              _cache
              (error: [k]b)
              : ([k]a, ()) =
    (map g error, ())

  let init 'a 'b (f: a -> b) (g: b -> a) : reshape_layer a b t =
    {forward  = \k -> forward k f,
     backward = \k -> backward k g,
     pickle = pickle.cst (),
     specs = [],
     functor = F.cst (),
     w_init  = \() -> (),
     update_weights = \() () -> ()
    }
}
