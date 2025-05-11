import "../nn_types"
import "layer_type"
import "dense"
--import "conv2d"
--import "flatten"
--import "max_pooling"


module layers (R: Real) = {

  type t = R.t

  module R = R

  module dense_layer   = dense R
  --module conv2d_layer  = conv2d R
  --module maxpool_layer = max_pooling_2d R
  --module flatten_layer = flatten R

  --type^ conv2d_tp [p][m][n] [filter_d] [filters] [out_m] [out_n] =
  --  conv2d_layer [p][m][n] [filter_d] [filters] [out_m] [out_n] t

  --type^ max_pooling_tp [nlayer] [input_m][input_n] [output_m][output_n] =
  --  max_pooling_2d_layer [nlayer] [input_m][input_n] [output_m][output_n] t

  let dense = dense_layer.init

  --let conv2d = conv2d_layer.init

  --let flatten = flatten_layer.init

  --let max_pooling2d = maxpool_layer.init

}
