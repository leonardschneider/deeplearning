import "../nn_types"

module type optimizer_type = {

  type t
  type ^learning_rate

  -- | Train function with signature
  --   network -> learning_rate -> input data -> labels
  --   -> batch_size -> classifier
  --   Returns the new network with updated weights and the training loss
  val train [K] 'i 'w 'g 'e2 'o [s] [ps]:
    NN i w o g o e2 (apply_grad3 t) [s] [ps] ->
    weights w ->
    learning_rate ->
    (input: [K]i) ->
    (labels: [K]o) ->
    (batch_size:i64) ->
    loss_func o t ->
    (weights w, t)
}
