import "../nn_types"
import "optimizer_type"
import "gradient_descent"


-- | Collection module for optimizers to be accessed through
module type optimizers =  {

  type t
  val gradient_descent [n][m][K] 'i 'w 'g 'o 'e2 [s] [ps]:
    NN ([n]i) w ([m]o) g ([m]o) e2 (apply_grad3 t) [s] [ps]
    -> weights w
    -> t
    -> ([K][n]i)
    -> ([K][m]o)
    -> i64
    -> loss_func ([m]o) t
    -> (weights w, t)
}

module optimizers_coll (R:real) : optimizers with t = R.t = {


  type t = R.t
  module gd = gradient_descent R

  let gradient_descent [n][m][K] 'w 'g 'o 'e2 'i [s] [ps]
                      (nn: NN ([n]i) w ([m]o) g ([m]o) e2 (apply_grad3 t) [s] [ps])
                      (ws: weights w)
                      (alpha:t)
                      (input: [K][n]i) (labels: [K][m]o) (step_sz: i64)
                      (loss: loss_func ([m]o) t) =
    gd.train nn ws alpha input labels step_sz loss

  let step_lr = \gamma (step_size: i64) epoch step lr ->
    lr * (gamma ** (step / step_size))

  let cst_lr = \_ _ lr -> lr

}

