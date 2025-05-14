import "../nn_types"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"
import "../../../leonardschneider/pickle/pickle"
import "../../../leonardschneider/functor/functor"

-- | Conv1d layer
-- Reference: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#conv1d
-- Notes: groups are not supported

-- Conv1d layer output type
--  - [L_in]: length of the input sequence
--  - [K]: kernel size
--  - [P]: padding
--  - [S]: stride
--  - [D]: dilation
type L_out [L_in][K][P][S][D] 't =
  [(P + L_in + P - D * (K - 1) - 1) // S + 1]t

-- | Conv1d layer type
--  - [C_in]: number of input channels
--  - [L_in]: length of the input sequence
--  - [C_out]: number of output channels
--  - [K]: kernel size
--  - [P]: padding
--  - [S]: stride
--  - [D]: dilation
type^ conv1d_layer [C_in][L_in][C_out][K][P][S][D] 't [p] [ts] =
  --  input             weights            output
  NN  ([C_in][L_in]t)  ([C_out][C_in][K]t, [C_out]t) ([C_out](L_out [L_in][K][P][S][D] t))
  --  cache
      ([C_in][P + L_in + P]t)
      t [] [p] [ts]

type padding_mode = #zeros | #reflect | #replicate | #circular

type padding = {
  sz: i64,
  mode: padding_mode
}

module conv1d (R: Real) = {

  type t            = R.t

  module lalg   = mk_linalg R

  let pad [n] (padding: padding) (x: [n]t): [padding.sz + n + padding.sz]t =
    let (before, after): ([padding.sz]t, [padding.sz]t) = match padding.mode
      case #zeros -> let p = replicate padding.sz (R.i32 0) in (p, p)
      case #reflect -> (
        let before = x |> take padding.sz |> reverse
        let after  = x |> reverse |> take padding.sz
        in (before, after)
      )
      case #replicate -> (
        let before = replicate padding.sz x[0]
        let after  = replicate padding.sz (last x)
        in (before, after)
      )
      case #circular -> (
        let before = x |> reverse |> take padding.sz |> reverse
        let after  = x |> take padding.sz
        in (before, after)
      )
    in before ++ x ++ after


  -- | Single convolution
  let conv1 [K] (L_in: i64) (P: i64) (S: i64) (D: i64) (ker: [K]t) (x: [P + L_in + P]t): (L_out [L_in][K][P][S][D] t) =
    iota ((P + L_in + P - D * (K-1) -1) // S + 1) -- number of strides
      |> map (\s ->
        -- kernel indices against the input
        iota K
          |> map (*D)          -- dilation
          |> map (+(s*S))      -- stride offset
          |> map (\i -> x[i])  -- inputs
          |> map2 (R.*) ker    -- kernel convolution
          |> R.sum
      )

  -- | Channel convolution
  let conv [C_in][C_out][K] (L_in: i64) (P: i64) (S: i64) (D: i64) (ker: [C_out][C_in][K]t) (b: [C_out]t) (x: [C_in][P + L_in + P]t): [C_out](L_out [L_in][K][P][S][D] t) =
    map2 (\k b ->
      map2 (conv1 L_in P S D) k x -- kernel convolution for all input channels
        |> transpose              -- transpose input channel and sequence length dimensions
        |> map R.sum              -- sum over input channels convolutions
        |> map (R.+b)             -- add bias for each output channel
    ) ker b

  -- | Backpropagation helper for a single kernel
  let deconv1_dx [K] (L_in: i64) (P: i64) (S: i64) (D: i64) (dy: (L_out [L_in][K][P][S][D] t)) (ker: [K]t): [P + L_in + P]t =
    iota (P + L_in + P - D * (K-1) -1) -- number of strides
      |> map (\s ->
        -- kernel indices against the input
        let is = iota K
          |> map (*D)          -- dilation
          |> map (+(s*S))      -- stride offset
        let vs = ker |> map (R.* dy[s])
        in spread (P + L_in + P) (R.i32 0) is vs
      )
      |> transpose
      |> map (R.sum) -- sum all kernel gradients

  let deconv1_dk [K] (L_in: i64) (P: i64) (S: i64) (D: i64) (x: [P + L_in + P]t) (dy: (L_out [L_in][K][P][S][D] t)): [K]t =
    tabulate (P + L_in + P - D * (K-1) -1) (\s -> -- number of strides
        -- kernel indices against the input
        iota K
          |> map (*D)          -- dilation
          |> map (+(s*S))      -- stride offset
          |> map (\i -> x[i])  -- inputs
          |> map (R.* dy[s]) -- kernel convolution
      )
      |> transpose
      |> map (R.sum) -- sum all inputs gradients

  -- | Backpropagation helper for all channel kernels
  let deconv_dx [C_in][C_out][K] (L_in: i64) (P: i64) (S: i64) (D: i64) (ker: [C_out][C_in][K]t) (dy: [C_out](L_out [L_in][K][P][S][D] t)): [C_in][P + L_in + P]t =
    map2 (\k dy -> map (deconv1_dx L_in P S D dy) k) ker dy
      |> transpose |> map (transpose)
      |> map (map (R.sum)) -- sum all kernel gradients

  let deconv_dk [C_in][C_out][K] (L_in: i64) (P: i64) (S: i64) (D: i64) (x: [C_in][P + L_in + P]t) (dy: [C_out](L_out [L_in][K][P][S][D] t)): [C_out][C_in][K]t =
    tabulate_2d C_out C_in (\c_in c_out -> deconv1_dk L_in P S D x[c_in] dy[c_out])

  -- Forward propagation
  -- Applies a 1D convolution over an input signal composed of several input planes.
  -- The input is a 3D tensor of shape (N, C_in, L_in) and the output is a 3D tensor of shape (N, C_out, L_out)
  let forward (N: i64)
              (C_in: i64) (L_in: i64)
              (C_out: i64) (K: i64)
              (padding: padding) (S: i64) (D: i64)
              (_training: bool)
              ((W, b): ([C_out][C_in][K]t, [C_out]t))
              (x: [N][C_in][L_in]t)
            : ([N][C_in][padding.sz + L_in + padding.sz]t, [N][C_out](L_out [L_in][K][padding.sz][S][D] t)) =
    let X = x |> map (map (pad padding))
    let y = X |> map (conv L_in padding.sz S D W b)
    in (X, y)

  -- Backward propagation
  let backward (N: i64)
                (C_in: i64) (L_in: i64)
                (C_out: i64) (K: i64)
                (padding: padding) (S: i64) (D: i64)
                (_training: bool)
                ((k, _): ([C_out][C_in][K]t, [C_out]t))
                (X: [N][C_in][padding.sz + L_in + padding.sz]t)
                (dy: [N][C_out](L_out [L_in][K][padding.sz][S][D] t))
              : ([N]([C_in][L_in]t), ([C_out][C_in][K]t, [C_out]t)) =
      let dx =
        dy
        |> map (deconv_dx L_in padding.sz S D k)
        |> map (map (drop padding.sz >-> take L_in))
      let dk =
        map2 (deconv_dk L_in padding.sz S D) X dy
        |> transpose |> map (transpose) |> map (map transpose)
        |> map (map (map R.sum))
      let db = dy |> transpose |> map (map R.sum) |> map R.sum
      in (dx, (dk, db))

  type config = {
    padding: padding,
    stride: i64,
    dilation: i64
  }

  let default: config = {
    padding = { sz = 0, mode = #zeros },
    stride  = 1,
    dilation = 1
  }

  module P = pickle
  module w_init = weight_initializer R

  let init [s] (label: [s]u8)
              (C_in: i64) (L_in: i64) (C_out: i64) (K: i64) (config: config) (seed: i32)
        : conv1d_layer [C_in][L_in][C_out][K][config.padding.sz][config.stride][config.dilation] t [C_out * (C_in * (K * R.sz)) + C_out * R.sz] [C_out * (C_in * (K * 1)) + C_out * 1] =
    let k = R.(sqrt ((i32 1) / ((i64 C_in) * (i64 K)))) in
    {
      forward  = \N -> forward  N C_in L_in C_out K config.padding config.stride config.dilation,
      backward = \N -> backward N C_in L_in C_out K config.padding config.stride config.dilation,
      pickle   = P.(pair (array C_out (array C_in (array K R.pu))) (array C_out R.pu)),
      specs    = label ++ ".weight" ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x03] ++ ((P.pickle P.i64) C_out) ++ ((P.pickle P.i64) C_in) ++ ((P.pickle P.i64) K)
              ++ label ++ ".bias"   ++ [0x00, 'b', 0x02] ++ R.tpe ++ [0x01] ++ ((P.pickle P.i64) C_out),
      functor  = F.(pair (array C_out (array C_in (array K scalar))) (array C_out scalar)),
      w_init     = \() -> (
        let ker = w_init.gen_random_array_uni (C_out*C_in*K) (R.neg k, k) seed |> unflatten_3d
        let b   = w_init.gen_random_array_uni C_out (R.neg k, k) seed
        in (ker, b)
      ),
      update_weights = \(w, b) (dw, db) ->
        (map2 (map2 (map2 (R.+))) w dw, map2 (R.+) b db),
    }
}