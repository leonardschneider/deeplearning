import "optimizer_type"
import "../nn_types"
import "../util"

-- | Stochastic gradient descent with momentum
module sgd_momentum (R: real): optimizer_type
  with t = R.t = {

  type t = R.t
  type state [n] = {
    lr: f32,
    b: [n]t,
    weight_decay: f32,
    nesterov: bool,
    momentum: f32,
    dampening: f32
  }

  let new_state [n] = \_ -> {
    lr = 0.001f32,
    b = replicate n R.(i32 0),
    weight_decay = 0.0f32,
    nesterov = false,
    momentum = 0.9f32,
    dampening = 0.0f32
  }

  let step 'w [n] = \lr (state: state [n]) theta grad ->
    let gamma: t = R.(f32 lr)
    let lambda = R.(f32 state.weight_decay)
    let mu = R.(f32 state.momentum)
    let tau = R.(f32 state.dampening)
    let nesterov = state.nesterov
    let b = state.b
    let grad = map2 (\g th -> g R.+ lambda R.* th) grad theta
    let b = map2 (\g b -> mu R.* b R.+ (R.(i32 1) R.- tau) R.* g) b grad
    let grad = if nesterov
               then map2 (\g b -> g R.+ mu R.* b) grad b
               else grad
    let theta = map2 (\th g -> th R.- gamma R.* (g)) theta grad
    let state = state with b = b
    in (theta, state)
  
  let new 'i 'o 'w 'c 'e_in 'e_out [s] [p] [n] (_: NN i w o c e_in e_out t [s] [p] [n]): Opt [n] t (state [n]) w = {
    step = step,
    new_state = new_state,
    lr = \(state: state [n]) -> state.lr
  }

}
