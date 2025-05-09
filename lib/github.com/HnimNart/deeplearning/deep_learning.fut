open import "nn_types"
open import "neural_network"
open import "layers/layers"
open import "loss_funcs"

-- | Aggregation module for deep learning
module deep_learning (R: Real) = {

  type t = R.t
  module nn     = neural_network R
  module layers = layers R
  module loss   = loss_funcs R

}
