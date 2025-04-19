-- | Module for creating random numbers
--   Used for weight initilization
import "../../diku-dk/cpprandom/random"

module weight_initializer (R:real) : {

  type t = R.t
  ---- Using xavier uniform initializer [-limit, limit]
  ---- with limit = sqrt(6/(fan_in + fan_out))
  val gen_random_array_2d_xavier_uni: (m:i64) -> (n:i64) -> i32 -> [n][m]t

  --- Using xavier norm initialize
  --- with mean = 0 and std = sqrt(2 / (fan_in + fan_out))
  val gen_random_array_2d_xavier_norm: (m: i64) -> (n: i64) -> i32 -> [n][m]t

} = {

  type t = R.t

  module norm = normal_distribution R minstd_rand
  module uni  = uniform_real_distribution R minstd_rand

  let gen_rand_uni (i:i32)  (dist:(uni.num.t, uni.num.t)): t =
    let rng = uni.engine.rng_from_seed [i]
    let (_, x) = uni.rand dist rng in x

  let gen_random_array_uni (d: i64) (dist) (seed: i32) : [d]t =
    map (\x -> gen_rand_uni x dist) (map (\x -> i32.i64 x + i32.i64 d + seed) (iota d))

  let gen_random_array_2d_xavier_uni (m: i64) (n: i64) (seed:i32) : [n][m]t =
    let d = R.(((sqrt((i32 6)) / sqrt(i64 n + i64 m))) )
    let arr = gen_random_array_uni (n*m) (R.(neg d),d) seed
    in unflatten arr

  let gen_rand_norm (i: i32) (dist) : t =
    let rng = norm.engine.rng_from_seed [i]
    let (_, x) = norm.rand dist rng in x

  let gen_random_array_norm (d: i64) (seed: i32) (dist) : [d]t =
    map (\x -> gen_rand_norm x dist) (map (\x -> i32.i64 x + i32.i64 d + seed) (iota d))

  let gen_random_array_2d_xavier_norm (m: i64) (n: i64) (seed:i32) : [n][m]t =
    let n_sqrt = R.(sqrt (i32 2/ (i64 m + i64 n)))
    let dist = {mean = R.(i32 0), stddev = n_sqrt }
    let arr = gen_random_array_norm (n*m) seed dist
    in unflatten (map (\x -> R.(x)) arr)
}
