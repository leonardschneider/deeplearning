import "optimizer_type"
import "../nn_types"
import "../util"

-- | Plain vanilla gradient descent optimizer
--   with mean gradient and constant learning rate
module sgd (R: real): optimizer_type
  with t = R.t = {

  type t = R.t
  type state [n] = {
    lr: f32,
    weight_decay: f32,
    witness: [n]()
  }

  let new_state [n] (): state [n] = {
    lr = 0.001,
    weight_decay = 0,
    witness = replicate n ()
  }

  let step 'w [n] = \lr (state: state [n]) theta grad ->
    let gamma: t = R.(f32 lr)
    let lambda = R.(f32 state.weight_decay)
    let grad = map2 (\g th -> g R.+ lambda R.* th) grad theta
    let theta = map2 (\th g -> th R.- gamma R.* (g)) theta grad
    in (theta, state)
  
  let new 'i 'o 'w 'c 'e_in 'e_out [s] [p] [n] (_: NN i w o c t [s] [p] [n]): Opt [n] t (state [n]) w = {
    step = step,
    new_state = new_state,
    lr = \(state: state [n]) -> state.lr
  }

}
