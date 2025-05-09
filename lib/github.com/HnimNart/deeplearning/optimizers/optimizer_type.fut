import "../nn_types"

type^ Opt [n] 't 's 'w = {
  step: (lr: f32) -> (state: s) -> (theta: [n]t) -> (grad: [n]t) -> ([n]t, s),
  new_state: () -> s,
  lr: s -> f32,
}

module type optimizer_type = {

  type t
  type state [n]

  -- | Create a new default state from the network weights
  val new 'i 'o 'w 'c 'e_in 'e_out [s] [p] [n]: NN i w o c e_in e_out t [s] [p] [n] -> Opt [n] t (state [n]) w

}

module trainer(R: real) = {

  type t = R.t

  -- | Train a network
  --   Returns the new state and the training loss
  let train [K] 'w 'g 'o 'e2 'i 'state [s] [ps] [ts]
            ({forward=f,
              backward=b,
              pickle=_,
              specs=_,
              functor=fun,
              w_init=_}: NN i w o g o e2 t [s] [ps] [ts])
            (ws: w)
            (optimizer: Opt [ts] t state w)
            (state: state)
            (lr: f32)
            (input: [K]i)
            (labels: [K]o)
            (batch_sz: i64)
            ({f=loss, fd=loss'}: loss_func o t) =

    let i = 0
    let training_loss = R.(i32 0)
    let (state', ws', training_loss', _) =
      loop (state, ws, training_loss, i) while i < K do
        let input'          = take batch_sz (drop i input)
        let label'          = take batch_sz (drop i labels)
        let (cache, output) = f batch_sz true ws input'
        let training_loss'  =
          map2 loss output label'
          |> reduce (R.+) R.(i32 0)
          |> (R.+) training_loss
        let error           = map2 (\o l -> loss' o l) output label'
        let (_, grads)      = b batch_sz false ws cache error
        let ws              = fun.flatten ws
        let grads           = fun.flatten grads
        let (ws', state')   = optimizer.step lr state ws grads
        let ws'             = fun.unflatten ws'
        in (state', ws', training_loss', i + batch_sz)
    in (state', { weights = ws'}, training_loss')

}
