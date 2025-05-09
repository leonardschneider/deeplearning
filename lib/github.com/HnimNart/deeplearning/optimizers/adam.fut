import "optimizer_type"
import "../nn_types"
import "../util"

-- | Plain vanilla gradient descent optimizer
--   with mean gradient and constant learning rate
module adam (R: real): optimizer_type
  with t = R.t = {

  type t = R.t
  type state [n] = {
    lr: f32,
    time: i64,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: [n]t,
    v: [n]t
  }

  let new_state [n] (): state [n] = {
    lr    = 0.001,
    time  = 0,
    beta1 = 0.9,
    beta2 = 0.999,
    eps   = 1e-8f32,
    weight_decay = 0.0,
    m = replicate n R.(i32 0),
    v = replicate n R.(i32 0)
  }

  let step 'w [n] = \lr (state: state [n]) theta grad ->
    let gamma: t = R.(f32 lr)
    let lambda = R.(f32 state.weight_decay)
    let beta1 = R.(f32 state.beta1)
    let beta2 = R.(f32 state.beta2)
    let eps = R.(f32 state.eps)
    let time = R.(i64 state.time)
    let m = state.m
    let v = state.v
    let grad = map2 (\g th -> g R.+ lambda R.* th) grad theta
    let m = map2 (\g m -> beta1 R.* m R.+ (R.(i32 1) R.- beta1) R.* g) grad m
    let v = map2 (\g v -> beta2 R.* v R.+ (R.(i32 1) R.- beta2) R.* g R.* g) grad v
    let m_hat = map (\m -> m R./ (R.(i32 1) R.- beta1 R.** time)) m
    let v_hat = map (\v -> v R./ (R.(i32 1) R.- beta2 R.** time)) v
    let theta = map3 (\th m v -> th R.- gamma R.* m R./ (R.sqrt v R.+ eps) ) theta m_hat v_hat
    let state = state
                with time = state.time + 1
                with m = m
                with v = v
    in (theta, state)
  
  let new 'i 'o 'w 'c 'e_in 'e_out [s] [p] [n] (_: NN i w o c e_in e_out t [s] [p] [n]): Opt [n] t (state [n]) w = {
    step = step,
    new_state = new_state,
    lr = \(state: state [n]) -> state.lr
  }

}
