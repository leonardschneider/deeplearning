import "../../leonardschneider/pickle/pickle"
import "../../leonardschneider/functor/functor"

-- | Network types
type^ forwards   'input 'w 'output 'cache [k] = bool -> w -> [k]input -> ([k]cache, [k]output)
type^ backwards  'c 'w 'err_in 'err_out [k] = bool -> w -> [k]c -> [k]err_in -> ([k]err_out, w)

type^ NN 'input 'w 'output 'c 'e_in 'e_out 't [s] [p] [ts] = {
  forward : (k: i64) -> forwards input w output c [k],
  backward: (k: i64) -> backwards c w e_in e_out [k],
  pickle: pickle.pu w [p],
  specs: [s]u8,
  functor: F.F t w [ts],
  w_init: () -> w
}

--- Commonly used types
type arr1d [n] 't = [n]t
type arr2d [n][m] 't = [n][m]t
type arr3d [n][m][p] 't = [n][m][p]t
type arr4d [n][m][p][q] 't = [n][m][p][q]t

type dims2d  = (i32, i32)
type dims3d  = (i32, i32, i32)

-- Opaque weight type to keep futhark-pycffi happy
type weights 't = { weights: t }

let get_weights 'w (ws: weights w): w = ws.weights

--- The 'standard' weight definition
--- used by optimizers
type std_weights [a][b][c] 't = ([a][b]t, [c]t)
type^ apply_grad2 'x 'y = (x, y) -> (x, y) -> (x, y)
type^ apply_grad3 't = (a: i64) -> (b: i64) -> apply_grad2 ([a][b]t) ([a]t)


--- Function pairs
--- Denotes a function and it's derivative
type^ activation_func 'o = {f: o -> o, fd: o -> o}
type^ loss_func 'o  't   = {f: o -> o -> t, fd: o -> o -> o}

--- Real numbers with serialization
module type Real = {
  include real
  val sz: i64
  type^ Pu = pickle.pu t [sz]
  val pu: Pu
  val tpe: [4]u8
}

module f32: Real with t = f32 = {
  open f32
  let sz = 4i64
  type^ Pu = pickle.pu f32 [sz]
  let pu = (pickle.f32 :> pickle.pu f32 [sz])
  let tpe = " f32"
}
